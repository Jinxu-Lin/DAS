import os
import math
import argparse
import logging
import random
import inspect
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_scheduler

from Tools.Dataloader import cifar2

dataset_loader = {
    'cifar2': cifar2,
}

def set_seeds(seed):
    set_seed(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model():
    model = UNet2DModel(
        sample_size=32,
        freq_shift=1,
        flip_sin_to_cos=False,
        down_block_types= (
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D"
        ),
        up_block_types= (
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D"
        ),
        block_out_channels=(128, 256, 256, 256),
        downsample_padding=0,
        attention_head_dim=None,
        norm_eps=1e-6,
        resnet_time_scale_shift='scale_shift',
    )
    return model


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # seed
    parser.add_argument("--seed", type=int, default=42,
                        dest="seed", help="random seed")
    
    # logging
    parser.add_argument("--logger", type=str, default='tensorboard',
                        dest="logger", choices=['tensorboard', 'wandb'],
                        help='logger')
    parser.add_argument("--logger-name", type=str, default='cifar2',
                        dest="logger_name", help='logger name')
    parser.add_argument("--logging-dir", type=str, default='logs',
                        dest="logging_dir", help='logging directory')

    # dataset
    parser.add_argument("--dataset", type=str, default="cifar2",
                        dest="dataset", help="dataset name")
    parser.add_argument("--load-dataset", action="store_true", default=False,
                        dest="load_dataset", help='load local dataset')
    parser.add_argument("--dataset-dir", type=str, default=None,
                        dest="dataset_dir", help='dataset directory')
    parser.add_argument("--index-path", type=str, default=None,
                        dest="index_path", help='index path')
    
    # data loader
    parser.add_argument("--data-aug", action="store_true", default=True,
                        dest="data_aug", help='data augmentation')
    parser.add_argument("--resolution", type=int, default=32,
                        dest="resolution", help='resolution of the dataset')
    parser.add_argument("--shuffle", action="store_true", default=False,
                        dest="shuffle", help='shuffle the dataset')
    parser.add_argument("--center-crop", action="store_true", default=False,
                        dest="center_crop", help='center crop the dataset')
    parser.add_argument("--random-flip", action="store_true", default=False,
                        dest="random_flip", help='random flip the dataset')
    parser.add_argument("--batch-size", type=int, default=64, 
                        dest="batch_size", help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0,
                        dest="dataloader_num_workers", help="The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    
    # ddpm
    parser.add_argument("--model-config", type=str, default="None",
                        dest="model_config", help="model config name or path")
    parser.add_argument("--prediction-type", type=str, default="epsilon",
                        dest="prediction_type", choices=["epsilon", "sample"],
                        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.")
    parser.add_argument("--ddpm-num-steps", type=int, default=1000,
                        dest="ddpm_num_steps", help="number of inference steps for DDPM")
    parser.add_argument("--ddpm-beta-schedule", type=str, default="linear",
                        dest="ddpm_beta_schedule", choices=["linear", "cosine"],
                        help="beta schedule for DDPM")
    
    # optimizer and learning rate scheduler
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        dest="learning_rate", help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--adam-beta1", type=float, default=0.95,
                        dest="adam_beta1", help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999,
                        dest="adam_beta2", help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=1e-6,
                        dest="adam_weight_decay", help="Weight decay magnitude for the Adam optimizer.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-8,
                        dest="adam_epsilon", help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        dest="lr_scheduler", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')

    # train
    parser.add_argument("--num-epochs", type=int, default=200,
                        dest="num_epochs", help="number of epochs")

    # save
    parser.add_argument("--save-dir", type=str, default='./saved',
                        dest="save_dir", help='save directory')
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        dest="resume_from_checkpoint", 
                        help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `latest` to automatically select the last available checkpoint."
    )
    parser.add_argument("--checkpointing-steps", type=int, default=500,
                        dest="checkpointing_steps", 
                        help="Save a checkpoint of the training state every X updates. "
                        "These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        dest="gradient_accumulation_steps",
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--save-model-epochs", type=int, default=10,
                        dest="save_model_epochs",
                        help="Save the model every X epochs.")

    args = parser.parse_args()
    return args


def main(args):

    # set seeds
    set_seeds(args.seed)

    # Initialize logger and accelerator
    logger = get_logger(__name__, log_level="INFO")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging_dir = os.path.join(args.save_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=args.logger,
        project_config=accelerator_project_config,
    )

    logger.info(accelerator.state, main_process_only=False)

    # Load dataset
    logger.info(f"Loading dataset {args.dataset}...")
    train_dataloader = dataset_loader[args.dataset].get_train_loader(
        args,
    )

    # Initialize model
    model = get_model()
        
    # Initialize the DDPM scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
    )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize the learning rate scheduler
    lr_warmup_steps = math.ceil((len(train_dataloader) * args.num_epochs)*0.1)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Checkpointing
    if args.checkpointing_steps==-1:
        args.checkpointing_steps=math.ceil((len(train_dataloader) * args.num_epochs)*0.01)
    
    # Prepare accelerator
    train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, model, optimizer, lr_scheduler
    )
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        print(args.logger_name)
        accelerator.init_trackers(args.logger_name, config=vars(args))

    # Training
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.save_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):    
        
        model.train()

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["input"]
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            # Number of images in this batch
            num_images = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (num_images,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):

                # Predict the noise residual
                model_output = model(noisy_images, timesteps).sample

                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output, noise)  # this could have different weights!
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    # use SNR weighting from distillation paper
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_images, reduction="none"
                    )  
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = f"{args.save_dir}/checkpoint-{global_step}"
                        os.makedirs(save_path, exist_ok=True)
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)
                unet.eval()

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                os.makedirs(args.save_dir, exist_ok=True)
                pipeline.save_pretrained(args.save_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)