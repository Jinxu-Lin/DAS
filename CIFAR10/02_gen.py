import argparse
import logging
import os
import random
import numpy as np
import torch
from accelerate.utils import set_seed

from diffusers import DDIMPipeline, DDPMScheduler, UNet2DModel
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
    parser = argparse.ArgumentParser(description="CIFAR2 Image Generation using Trained Diffusion Model")
    
    # Core parameters
    parser.add_argument("--model_config_name_or_path", type=str, required=True, help="UNet model config path")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--save_path", type=str, required=True, help="Save Path for generated images")
    
    # Generation parameters
    parser.add_argument("--train_batch_size", type=int, default=256, help="Batch size for generation")
    parser.add_argument("--gen_seed", type=int, default=0, help="Seed for image generation")
    parser.add_argument("--num_images", type=int, default=1000, help="Total number of images to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of DDIM inference steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter (0.0 for deterministic)")
    
    # System parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
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


def load_pretrained_pipeline(model_path, logger):
    """Load pretrained model using the same method as training"""
    
    logger.info(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load the pipeline that was saved by train.py
    pipeline = DDIMPipeline.from_pretrained(model_path)
    pipeline.cuda()
    
    logger.info("Model loaded and moved to GPU")
    return pipeline


def generate_images(pipeline, args, logger):
    """Generate images using the pipeline"""
    
    logger.info(f"Generating {args.num_images} images with batch size {args.train_batch_size}")
    logger.info(f"Inference steps: {args.num_inference_steps}, eta: {args.eta}")
    
    # Create save path
    os.makedirs(args.save_path, exist_ok=True)
    
    total_generated = 0
    
    for i in range(0, args.num_images, args.train_batch_size):
        # Calculate batch size for this iteration
        bsz = min(args.train_batch_size, args.num_images - i)
        
        logger.info(f"Generating batch {i//args.train_batch_size + 1}, "
                   f"images {i} to {i + bsz - 1}")
        
        # Create generators with different seeds for each image
        generators = [
            torch.Generator('cpu').manual_seed(args.gen_seed * args.num_images + i + j) 
            for j in range(bsz)
        ]
        
        # Generate images
        with torch.no_grad():
            images = pipeline(
                generator=generators,
                batch_size=bsz,
                num_inference_steps=args.num_inference_steps,
                output_type="numpy",
                eta=args.eta,
            ).images
        
        # Convert to PIL and save
        images_pil = pipeline.numpy_to_pil(images)
        
        for idx, image in enumerate(images_pil):
            image_path = os.path.join(args.save_path, f'{i + idx}.png')
            image.save(image_path)
            total_generated += 1
        
        logger.info(f"Saved {len(images_pil)} images, total: {total_generated}/{args.num_images}")
    
    logger.info(f"Generation complete! Saved {total_generated} images to {args.save_path}")


def main():
    args = parse_args()

    logger = logging.getLogger(__name__)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f"Arguments: {args}")
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    # Load the complete pipeline (model + scheduler) that was saved by train.py
    pipeline = load_pretrained_pipeline(args.model_path, logger)
    
    # Generate images
    generate_images(pipeline, args, logger)


if __name__ == "__main__":
    main()