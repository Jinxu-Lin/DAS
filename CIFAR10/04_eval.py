import argparse
import logging
import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, Image
from torchvision import transforms

from diffusers import DDPMScheduler, UNet2DModel
from diffusers.utils import check_min_version

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
    parser = argparse.ArgumentParser(description="CIFAR2 Model Evaluation - Loss Computation on Single Model")
    
    # Core parameters
    parser.add_argument("--model_config_name_or_path", type=str, required=True, help="UNet model config path")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the specific model to evaluate")
    parser.add_argument("--dataset_name_or_path", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset (train, val, gen)")
    parser.add_argument("--index_path", type=str, required=True, help="Path to data indices")
    parser.add_argument("--gen_path", type=str, required=True, help="Path of generated images")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save features")
    
    # Data parameters
    parser.add_argument("--resolution", type=int, default=32, help="Input image resolution")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop images")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="Number of dataloader workers")
    
    # Evaluation parameters
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of timesteps to evaluate")
    
    # Seeds
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--e_seed", type=int, default=0, help="Evaluation seed for noise generation")
    
    args = parser.parse_args()
    return args


def create_model(config_path, logger):
    """Create UNet2D model from config"""
    logger.info(f"Loading model config from: {config_path}")

    config = UNet2DModel.load_config(config_path)
    config['resnet_time_scale_shift'] = 'scale_shift'
    model = UNet2DModel.from_config(config)

    logger.info(f"Successfully created model")
    
    return model


def load_pretrained_model(model_path, logger):
    """Load pretrained UNet2DModel from diffusers format (supports safetensors)"""
    logger.info(f"Loading UNet2DModel from: {model_path}/unet")

    model_path = os.path.join(model_path, "unet")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    model = UNet2DModel.from_pretrained(model_path)
    model.cuda()
    model.eval()
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


def load_gen_dataset(args, logger):
    """Load generated dataset"""

    if not os.path.isdir(args.gen_path):
        raise FileNotFoundError(f"Generated images path not found: {args.gen_path}")
    
    image_files = glob.glob(os.path.join(args.gen_path, "*.png"))
    image_files.sort()
    logger.info(f"Found {len(image_files)} image files in {args.gen_path}")

    
    dataset = DatasetDict({
        "gen": Dataset.from_dict({
            "img": image_files,
        }).cast_column("img", Image()),
    })
    dataset = dataset["gen"]

    return dataset


def load_dataset(args, logger):
    """Load dataset based on index path type"""
    logger.info(f"Loading {args.dataset_type} dataset")

    if args.dataset_type == "train":
        dataset = load_train_dataset(args, logger)
        logger.info(f"Successfully loaded training dataset with {len(dataset)} samples)")
        
    elif args.dataset_type == "val":
        # Validation dataset
        dataset = load_train_dataset(args.dataset_name_or_path, split="test")
        logger.info(f"Successfully loaded validation dataset with {len(dataset)} samples")
        
    elif args.dataset_type == "gen":
        # Generated images dataset
        dataset = load_gen_dataset(args, logger)
        logger.info(f"Successfully loaded generated dataset with {len(dataset)} samples")
    
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    return dataset


def select_dataset_from_index(dataset, args, logger):
    """Select dataset by subset indices"""
    
    logger.info(f"Loading indices from {args.index_path}")
    with open(args.index_path, 'rb') as handle:
        sub_idx = pickle.load(handle)
    
    dataset = dataset.select(sub_idx)
    logger.info(f"Dataset size after filtering: {len(dataset)}")


def create_dataloader(dataset, args, logger):
    """Create dataloader with transforms"""
    logger.info(f"Creating training dataloader")
    
    # Data transforms
    augmentations = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}
    
    dataset.set_transform(transform_images)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    return dataloader


def compute_losses_for_model(model, dataloader, noise_scheduler, args, logger):
    """Compute losses for all batches and timesteps"""
    logger.info(f"Computing losses across {args.num_timesteps} timesteps")
    
    batch_loss_list = []
    
    for step, batch in enumerate(dataloader):
        logger.info(f"Processing batch {step + 1}/{len(dataloader)}")
        
        # Move batch to GPU
        for key in batch.keys():
            batch[key] = batch[key].cuda()
        
        latents = batch["input"]
        bsz = latents.shape[0]
        
        # Compute losses across for selected timesteps
        time_loss_list = []
        timestep_interval = max(1, 1000 // args.num_timesteps)
        
        for index_t, t in enumerate(range(0, 1000, timestep_interval)):

            if index_t % 100 == 0:
                logger.info(f"Processing timestep {index_t}/{args.num_timesteps}")

            # Create timesteps tensor
            timesteps = torch.tensor([t] * bsz, device=latents.device).long()
            
            # Set seed for reproducible noise generation
            set_seeds(args.e_seed * 1000 + t)
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get target based on prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            # Compute loss
            with torch.no_grad():
                model_pred = model(noisy_latents, timesteps).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                # Average over spatial dimensions, keep batch dimension
                loss = loss.mean(dim=list(range(1, len(loss.shape))))
                time_loss_list.append(loss.detach().cpu().numpy())
        
        batch_loss_list.append(time_loss_list)
    
    return batch_loss_list


def save_results(batch_loss_list, args, logger):
    """Save evaluation results to pickle file"""
    output_file = f'{args.save_path}/ddpm-{args.dataset_type}-errors-{args.num_timesteps}.pkl'
    logger.info(f"Saving {args.dataset_type} results to: {output_file}")
    
    # Ensure Save Path exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    with open(output_file, 'wb') as handle:
        pickle.dump(batch_loss_list, handle)
    logger.info(f"Successfully saved results to {output_file}")


def evaluate_model(model, dataloader, noise_scheduler, args, logger):
    """Evaluate a single model"""
    
    # Compute losses
    batch_loss_list = compute_losses_for_model(model, dataloader, noise_scheduler, args, logger)
    
    # Save results
    save_results(batch_loss_list, args, logger)
    
    logger.info(f"Completed evaluation for model")


def main():
    args = parse_args()
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f"Arguments: {args}")

    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    logger.info(f"Starting evaluation on {args.dataset_type} dataset")
    
    # Create model
    model = create_model(args.model_config_name_or_path, logger)
    model = load_pretrained_model(args.model_path, logger)
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
    # Create dataset and dataloader
    dataset = load_dataset(args, logger)
    if args.dataset_type == "train" or args.dataset_type == "val":
        dataset = select_dataset_from_index(dataset, args, logger)
    dataloader = create_dataloader(dataset, args, logger)
    
    # Evaluate single model
    evaluate_model(model, dataloader, noise_scheduler, args, logger)
    
    logger.info(f"Completed evaluation on {args.dataset_type} dataset")


if __name__ == "__main__":
    main()