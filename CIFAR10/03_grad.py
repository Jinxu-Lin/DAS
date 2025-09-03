import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import pickle
import glob

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict, Dataset, Image

from diffusers import DDPMScheduler, UNet2DModel
from diffusers.utils import check_min_version

from trak.projectors import ProjectionType, CudaProjector

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
    parser = argparse.ArgumentParser(description="CIFAR2 Gradient Computation for Diffusion Model")
    
    # Core parameters
    parser.add_argument("--model_config_name_or_path", type=str, required=True, help="UNet model config path")
    parser.add_argument("--model_path", type=str, required=True, help="Output directory containing trained model")
    parser.add_argument("--dataset_name_or_path", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset (train, val, gen)")
    parser.add_argument("--index_path", type=str, required=True, help="Path to data indices")
    parser.add_argument("--gen_path", type=str, required=True, help="Path to generated images")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save features")
    
    # Data parameters
    parser.add_argument("--resolution", type=int, default=32, help="Input image resolution")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop images")
    parser.add_argument("--train_batch_size", type=int, default=52, help="Batch size for processing")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers")
    
    # Gradient computation parameters
    parser.add_argument("--split", type=int, required=True, help="Data split index")
    parser.add_argument("--loss_function_type", type=str, required=True, help="Loss function type (mse, weighted-mse, l1-norm, l2-norm, etc.)")
    parser.add_argument("--projection_dim", type=int, required=True, help="Projection dimension for gradient compression")
    parser.add_argument("--num_timesteps_avg", type=int, required=True, help="Number of timesteps for averaging gradients")    
    parser.add_argument("--timestep_strategy", type=str, required=True, help="Timestep sampling strategy (uniform, cumulative)")
    
    # Seeds
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--e_seed", type=int, default=0, help="Evaluation seed")
    
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
        
        # Select subset based on split
        start_idx = args.split * 10000
        end_idx = (args.split + 1) * 10000
        dataset = dataset.select(range(start_idx, end_idx))
        
        logger.info(f"Successfully loaded training dataset with {len(dataset)} samples by split {args.split}")
        
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


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_projector(model, args, logger):
    """Setup gradient projector"""
    
    grad_dim = count_parameters(model)
    projector = CudaProjector(
        grad_dim=grad_dim,
        proj_dim=args.projection_dim,
        seed=42,
        proj_type=ProjectionType.normal,
        device='cuda:0'
    )
    
    logger.info(f"Model parameters: {grad_dim//1e6:.2f}M")
    logger.info(f"Projection dimension: {args.projection_dim}")
    
    return projector


def create_loss_function(args, noise_scheduler, model, logger):
    """Create loss function based on args.f"""
    # Precompute weights for weighted loss functions
    weights = noise_scheduler.betas / (2 * noise_scheduler.alphas * (1 - noise_scheduler.alphas_cumprod))
    
    from torch.func import functional_call
    
    def compute_base_loss(params, buffers, noisy_latents, timesteps, targets, loss_type):
        """Base function for computing different loss types"""
        noisy_latents = noisy_latents.unsqueeze(0)
        timesteps = timesteps.unsqueeze(0)
        targets = targets.unsqueeze(0)
        
        predictions = functional_call(model, (params, buffers), args=noisy_latents,
                                    kwargs={'timestep': timesteps})
        predictions = predictions.sample.float()
        
        if loss_type == 'mean-squared-l2-norm':
            f = F.mse_loss(predictions, torch.zeros_like(targets).float(), reduction="none")
            f = f.reshape(1, -1).mean()
            
        elif loss_type == 'mean':
            f = predictions.reshape(1, -1).mean()
            
        elif loss_type == 'l1-norm':
            f = torch.norm(predictions.reshape(1, -1), p=1.0, dim=-1).mean()
            
        elif loss_type == 'l2-norm':
            f = torch.norm(predictions.reshape(1, -1), p=2.0, dim=-1).mean()
            
        elif loss_type == 'linf-norm':
            f = torch.norm(predictions.reshape(1, -1), p=float('inf'), dim=-1).mean()
            
        elif loss_type == 'weighted-mse':
            w = weights.to(device=timesteps.device)[timesteps]
            f = w * F.mse_loss(predictions, targets.float(), reduction="none")
            f = f.reshape(1, -1).mean()
            
        else:  # default MSE
            f = F.mse_loss(predictions, targets.float(), reduction="none")
            f = f.reshape(1, -1).mean()
        
        return f
    
    def compute_f(params, buffers, noisy_latents, timesteps, targets):
        return compute_base_loss(params, buffers, noisy_latents, timesteps, targets, args.loss_function_type)
    
    logger.info(f"Using loss function: {args.loss_function_type}")
    return compute_f


def get_timestep_strategy(args, logger):
    """Get timestep sampling strategy"""
    if args.timestep_strategy == 'uniform':
        selected_timesteps = range(0, 1000, 1000 // args.num_timesteps_avg)
    elif args.timestep_strategy == 'cumulative':
        selected_timesteps = range(0, args.num_timesteps_avg)
    else:
        raise ValueError(f"Unknown timestep strategy: {args.timestep_strategy}")
    
    logger.info(f"Timestep strategy: {args.timestep_strategy}, num_timesteps_avg={args.num_timesteps_avg}")
    logger.info(f"Selected timesteps: {list(selected_timesteps)}")
    
    return selected_timesteps


def vectorize_and_ignore_buffers(g):
    """Vectorize gradients and flatten them"""
    batch_size = len(g[0])
    out = []
    for b in range(batch_size):
        out.append(torch.cat([x[b].flatten() for x in g]))
    return torch.stack(out)


def compute_gradients(model, dataloader, noise_scheduler, projector, args, logger):
    """Main gradient computation function"""
    # Setup model parameters
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers() if v.requires_grad}
    
    # Remove scaled_dot_product_attention to avoid conflicts
    if hasattr(F, "scaled_dot_product_attention"):
        delattr(F, "scaled_dot_product_attention")
    
    # Setup gradient computation
    from torch.func import grad, vmap
    
    compute_f = create_loss_function(args, noise_scheduler, model)
    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0))
    
    # Get timestep strategy
    selected_timesteps = get_timestep_strategy(args)
    
    # Setup output file
    if "idx-train.pkl" in args.index_path:
        split_type = "train"
    elif "idx-val.pkl" in args.index_path:
        split_type = "val"
    else:
        split_type = "gen"
    
    filename = os.path.join(
        f'{args.save_path}/features-{args.e_seed}',
        f'ddpm-{split_type}-keys-{args.split}-{args.loss_function_type}-{args.num_timesteps_avg}-{args.projection_dim}-{args.timestep_strategy}.npy'
    )
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create memory-mapped array for storing features
    dstore_keys = np.memmap(
        filename,
        dtype=np.float32,
        mode='w+',
        shape=(len(dataloader.dataset), args.projection_dim)
    )
    
    logger.info(f"Computing gradients for {len(dataloader.dataset)} samples")
    logger.info(f"Output file: {filename}")
    
    # Process batches
    for step, batch in enumerate(dataloader):
        set_seeds(42)
        
        # Move batch to GPU
        for key in batch.keys():
            batch[key] = batch[key].cuda()
        
        latents = batch["input"]
        bsz = latents.shape[0]
        
        # Compute gradients for selected timesteps
        emb = None
        for index_t, t in enumerate(selected_timesteps):
            timesteps = torch.tensor([t] * bsz, device=latents.device).long()
            
            # Set seed for noise generation
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
            
            # Compute per-sample gradients
            ft_per_sample_grads = ft_compute_sample_grad(
                params, buffers, noisy_latents, timesteps, target
            )
            ft_per_sample_grads = vectorize_and_ignore_buffers(list(ft_per_sample_grads.values()))

            grad_norms = torch.norm(ft_per_sample_grads, dim=-1, keepdim=True)
            normalized_ft_per_sample_grads = ft_per_sample_grads / (grad_norms + 1e-8)
            
            # Accumulate gradients
            if emb is None:
                emb = normalized_ft_per_sample_grads
            else:
                emb += normalized_ft_per_sample_grads
        
        # Average over timesteps and project
        emb = emb / args.num_timesteps_avg
        emb = projector.project(emb, model_id=0)
        
        # Save to memory-mapped array
        start_idx = step * args.train_batch_size
        end_idx = start_idx + bsz
        
        # Ensure successful write
        while np.abs(dstore_keys[start_idx:end_idx, :32]).sum() == 0:
            dstore_keys[start_idx:end_idx] = emb.detach().cpu().numpy()
        
        logger.info(f"Processed batch {step+1}/{len(dataloader)}, samples {start_idx}-{end_idx}")
    
    logger.info("Gradient computation completed successfully")


def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f"Arguments: {args}")
    
    # Create and load model
    model = create_model(args.model_config_name_or_path, logger)
    model = load_pretrained_model(model, args.model_path, logger)
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
    # Create dataset and dataloader
    dataset = load_dataset(args, logger)
    if args.dataset_type == "train" or args.dataset_type == "val":
        dataset = select_dataset_from_index(dataset, args, logger)
    dataloader = create_dataloader(dataset, args, logger)
    
    # Setup projector
    projector = setup_projector(model, args, logger)
    
    # Compute gradients
    compute_gradients(model, dataloader, noise_scheduler, projector, args, logger)


if __name__ == "__main__":
    main()