import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from datasets import load_dataset, DatasetDict, Dataset, Image
from torchvision import transforms
import pandas as pd
import pickle

from diffusers import DDPMScheduler, UNet2DModel
from diffusers.utils import check_min_version

# Will error if the minimal version of diffusers is not installed
check_min_version("0.16.0")

logger = logging.getLogger(__name__)


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
    parser.add_argument("--dataset_name", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--model_config_name_or_path", type=str, required=True, help="UNet model config path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory containing trained model")
    parser.add_argument("--index_path", type=str, required=True, help="Path to data indices")
    parser.add_argument("--gen_path", type=str, required=True, help="Path to generated images")
    
    # Data parameters
    parser.add_argument("--resolution", type=int, default=32, help="Input image resolution")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop images")
    parser.add_argument("--train_batch_size", type=int, default=52, help="Batch size for processing")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers")
    
    # Gradient computation parameters
    parser.add_argument("--split", type=int, required=True, help="Data split index")
    parser.add_argument("--num_timesteps_avg", type=int, required=True, help="Number of timesteps for averaging gradients")
    parser.add_argument("--projection_dim", type=int, required=True, help="Projection dimension for gradient compression")
    parser.add_argument("--loss_function_type", type=str, required=True, help="Loss function type (mse, weighted-mse, l1-norm, l2-norm, etc.)")
    parser.add_argument("--timestep_strategy", type=str, required=True, help="Timestep sampling strategy (uniform, cumulative)")
    
    # Seeds
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--e_seed", type=int, default=0, help="Evaluation seed")
    
    args = parser.parse_args()
    return args


def create_model(config_path):
    """Create and configure UNet2D model"""
    config = UNet2DModel.load_config(config_path)
    config['resnet_time_scale_shift'] = 'scale_shift'
    model = UNet2DModel.from_config(config)
    return model


def load_pretrained_model(model, model_path):
    """Load pretrained model weights"""
    if 'checkpoint-0' in model_path:
        logger.info(f"Loading from checkpoint: {model_path}")
    else:
        weight_path = f'{model_path}/unet/diffusion_pytorch_model.bin'
        logger.info(f"Loading weights from: {weight_path}")
        model.load_state_dict(torch.load(weight_path))
    
    model.cuda()
    model.eval()
    return model


def create_dataset(args):
    """Create dataset based on index path type"""
    if "idx-train.pkl" in args.index_path:
        # Training dataset
        dataset = load_dataset(args.dataset_name, split="train")
        
        with open(args.index_path, 'rb') as handle:
            sub_idx = pickle.load(handle)
        
        # Select subset based on split
        start_idx = args.split * 1000
        end_idx = (args.split + 1) * 1000
        sub_idx = sub_idx[start_idx:end_idx]
        dataset = dataset.select(sub_idx)
        
        logger.info(f"Training dataset: {len(dataset)} samples from split {args.split}")
        
    elif "idx-val.pkl" in args.index_path:
        # Validation dataset
        dataset = load_dataset(args.dataset_name, split="test")
        
        with open(args.index_path, 'rb') as handle:
            sub_idx = pickle.load(handle)
        
        # Select subset based on split
        start_idx = args.split * 1000
        end_idx = (args.split + 1) * 1000
        sub_idx = sub_idx[start_idx:end_idx]
        dataset = dataset.select(sub_idx)
        
        logger.info(f"Validation dataset: {len(dataset)} samples from split {args.split}")
        
    else:
        # Generated images dataset
        df = pd.DataFrame()
        df['path'] = [f'{args.gen_path}/{i}.png' for i in range(1000)]
        
        dataset = DatasetDict({
            "train": Dataset.from_dict({
                "img": df['path'].tolist(),
            }).cast_column("img", Image()),
        })
        dataset = dataset["train"]
        
        logger.info(f"Generated dataset: {len(dataset)} samples")
    
    return dataset


def create_dataloader(dataset, args):
    """Create dataloader with transforms"""
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


def setup_projector(model, args):
    """Setup gradient projector"""
    from trak.projectors import ProjectionType, CudaProjector
    
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


def create_loss_function(args, noise_scheduler, model):
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


def get_timestep_strategy(args):
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


def compute_gradients(model, dataloader, noise_scheduler, projector, args):
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
        f'{args.output_dir}/features-{args.e_seed}',
        f'ddpm-{split_type}-keys-{args.split}-{args.num_timesteps_avg}-{args.projection_dim}-{args.loss_function_type}-{args.timestep_strategy}.npy'
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
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f"Arguments: {args}")
    
    # Set seeds
    set_seeds(args.seed)
    
    # Create and load model
    model = create_model(args.model_config_name_or_path)
    model = load_pretrained_model(model, args.output_dir)
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
    # Create dataset and dataloader
    dataset = create_dataset(args)
    dataloader = create_dataloader(dataset, args)
    
    # Setup projector
    projector = setup_projector(model, args)
    
    # Compute gradients
    compute_gradients(model, dataloader, noise_scheduler, projector, args)


if __name__ == "__main__":
    main()