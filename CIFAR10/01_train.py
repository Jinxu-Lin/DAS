import argparse
import logging
import math
import os
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

import wandb

# Will error if the minimal version of diffusers is not installed
check_min_version("0.16.0")


def set_seeds(seed):
    """Set seeds for reproducibility"""
    
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR2 Diffusion Model Training")
    
    # Core training parameters
    parser.add_argument("--dataset_name_or_path", type=str, default="cifar10", help="Dataset name or path to local dataset")
    parser.add_argument("--model_config_name_or_path", type=str, required=True, help="UNet model config path")
    parser.add_argument("--save_path", type=str, required=True, help="Output Path for model and checkpoints")
    parser.add_argument("--index_path", type=str, required=True, help="Path to training indices pickle file")
    
    # Data parameters
    parser.add_argument("--resolution", type=int, default=32, help="Input image resolution")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop images")
    parser.add_argument("--random_flip", action="store_true", help="Whether to randomly flip images horizontally")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6, help="Adam weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpointing and logging
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--save_images_epochs", type=int, default=20, help="Save sample images every X epochs")
    parser.add_argument("--wandb_name", type=str, required=True, help="Wandb experiment name")
    
    # Fixed parameters (remove from args since they're constant)
    parser.add_argument("--logger", type=str, default="wandb", help="Logger type")
    parser.add_argument("--mixed_precision", type=str, default="no", help="Mixed precision training")
    
    args = parser.parse_args()
    return args


def create_model(config_path, logger):
    """Create UNet2D model from config"""
    
    logger.info(f"Creating model from config: {config_path}")
    
    config = UNet2DModel.load_config(config_path)
    config['resnet_time_scale_shift'] = 'scale_shift'
    model = UNet2DModel.from_config(config)
    
    logger.info(f"Successfully created model")
    
    return model


def model_dropout(model, logger):
    """Set dropout to 0.1 for all dropout layers"""

    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.1
            
            logger.info(f"Set dropout for {n}: {m.p}")
    
    return model


def load_train_dataset(args, logger):
    """Load training dataset"""

    if os.path.exists(args.dataset_name_or_path):
        logger.info(f"Loading local dataset from path: {args.dataset_name_or_path}")
        dataset = load_from_disk(os.path.join(args.dataset_name_or_path, "train"))
    else:
        logger.info(f"Loading online dataset by name: {args.dataset_name_or_path}")
        dataset = load_dataset(args.dataset_name_or_path, split="train")
    
    return dataset


def select_dataset_from_index(dataset, args, logger):
    """Select dataset by subset indices"""
    
    logger.info(f"Loading indices from {args.index_path}")
    with open(args.index_path, 'rb') as handle:
        sub_idx = pickle.load(handle)
    
    dataset = dataset.select(sub_idx)
    logger.info(f"Dataset size after filtering: {len(dataset)}")


def create_dataloader(dataset,args, logger):
    """Create dataloader"""
    
    logger.info(f"Creating dataloader")
    
    # Data transforms
    augmentations = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}
    
    dataset.set_transform(transform_images)
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.dataloader_num_workers
    )
    
    logger.info(f"Successfully created dataloader")
    
    return train_dataloader


def setup_training_components(model, args, train_dataloader, logger):
    """Setup optimizer, scheduler, and other training components"""
    
    # Initialize the scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.95, 0.999),
        weight_decay=args.adam_weight_decay,
        eps=1e-08,
    )
    
    # Calculate warmup steps (10% of total training steps)
    total_training_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = math.ceil(total_training_steps * 0.1)
    
    # Dynamic checkpointing steps if -1
    if args.checkpointing_steps == -1:
        args.checkpointing_steps = math.ceil(total_training_steps * 0.01)
    
    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=total_training_steps,
    )
    
    logger.info(f"Total training steps: {total_training_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Checkpointing every {args.checkpointing_steps} steps")
    
    return noise_scheduler, optimizer, lr_scheduler


def train_epoch(model, train_dataloader, noise_scheduler, optimizer, lr_scheduler, 
                accelerator, args, epoch, global_step, logger):
    """Train for one epoch"""
    model.train()
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["input"]
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bsz = clean_images.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
        ).long()
        
        # Add noise to images (forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        if step == 0:
            logger.info(f"Noisy images dtype: {noisy_images.dtype}")
        
        with accelerator.accumulate(model):
            # Predict the noise residual
            model_output = model(noisy_images, timesteps).sample
            loss = F.mse_loss(model_output, noise)
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Update progress and logging
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            
            # Save checkpoint
            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.save_path, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
        
        # Log metrics
        logs = {
            "loss": loss.detach().item(), 
            "lr": lr_scheduler.get_last_lr()[0], 
            "step": global_step
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
    
    progress_bar.close()
    return global_step


def generate_and_save_samples(model, noise_scheduler, accelerator, args, epoch, global_step):
    """Generate and save sample images"""
    if not accelerator.is_main_process:
        return
        
    if epoch % args.save_images_epochs != 0 and epoch != args.num_epochs - 1:
        return
    
    unet = accelerator.unwrap_model(model)
    unet.eval()
    
    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
    generator = torch.Generator(device=pipeline.device).manual_seed(42)
    
    # Generate sample images
    images = pipeline(
        generator=generator,
        batch_size=16,  # Fixed eval batch size
        num_inference_steps=1000,  # Fixed inference steps
        output_type="numpy",
    ).images
    
    # Log to wandb
    images_processed = (images * 255).round().astype("uint8")
    
    accelerator.get_tracker("wandb").log(
        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
        step=global_step,
    )


def save_model(model, noise_scheduler, accelerator, args, logger):
    """Save the final model"""
    if not accelerator.is_main_process:
        return
        
    unet = accelerator.unwrap_model(model)
    unet.eval()
    
    pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
    pipeline.save_pretrained(args.save_path)
    logger.info(f"Model saved to {args.save_path}")


def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(total_limit=None)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
    )
    
    # Set logger
    logger = get_logger(__name__, log_level="INFO")
    logger.info(f"Training arguments: {args}")
    
    # Check wandb availability
    if not is_wandb_available():
        raise ImportError("Make sure to install wandb for logging during training.")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Create save path
    if accelerator.is_main_process:
        os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize model
    model = create_model(args.model_config_name_or_path)
    
    # Load training dataset
    dataset = load_train_dataset(args, logger)
    # Select training set by indices
    dataset = select_dataset_from_index(dataset, args, logger)
    # Create dataloader
    train_dataloader = create_dataloader(dataset, args, logger)
    
    # Setup training components
    noise_scheduler, optimizer, lr_scheduler = setup_training_components(model, args, train_dataloader)
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Initialize wandb tracking
    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_name, config=vars(args))
    
    # Training info
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        global_step = train_epoch(
            model, train_dataloader, noise_scheduler, optimizer, lr_scheduler,
            accelerator, args, epoch, global_step, logger
        )
        
        accelerator.wait_for_everyone()
        
        # Generate sample images every `save_images_epochs`
        # Comment if not needed
        # generate_and_save_samples(model, noise_scheduler, accelerator, args, epoch, global_step)
        
        # Save model at the end
        if epoch == args.num_epochs - 1:
            save_model(model, noise_scheduler, accelerator, args, logger)
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
