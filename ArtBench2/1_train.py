import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

from artbench2 import get_train_loader

def set_seeds(seed):
    set_seed(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # seed
    parser.add_argument("--seed", type=int, default=42,
                        dest="seed", help="random seed")

    # logging
    parser.add_argument("--logger", type=str, default="tensorboard",
                        dest="logger", help="The integration to report the results and logs to. Supported platforms are `tensorboard`, `wandb` and `comet_ml`. Use `all` to report to all integrations.")
    parser.add_argument("--logging-dir", type=str, default="logs",
                        dest="logging_dir", help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *save_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--wandb-name", type=str, default=None,
                        dest="wandb_name", help="the project name for wandb")
    
    # dataset
    parser.add_argument("--dataset-dir", type=str, default=None,
                        dest="dataset_dir", help="The directory of the dataset.")
    parser.add_argument("--index-path", type=str, default=None,
                        dest="index_path", help='index path')
    
    # dataloader
    parser.add_argument("--resolution", type=int, default=256,
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

    # stable diffusion
    parser.add_argument("--model-path", type=str, default=None,
                        dest="model_path", required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None,
                        dest="revision", required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    
    # optimizer and learning rate scheduler
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        dest="learning_rate", help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr-warmup-steps", type=int, default=500,
                        dest="lr_warmup_steps", help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam-beta1", type=float, default=0.9,
                        dest="adam_beta1", help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, 
                        dest="adam_beta2", help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2, 
                        dest="adam_weight_decay", help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, 
                        dest="adam_epsilon", help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, 
                        dest="max_grad_norm", help="Max gradient norm.")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        dest="lr_scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    
    # train
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                        dest="mixed_precision", choices=["None", "fp16", "bf16"],
                        help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                        " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                        " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."))
    parser.add_argument("--num-train-epochs", type=int, default=100,
                        dest="num_train_epochs", help="number of epochs")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        dest="gradient_accumulation_steps", help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max-train-steps", type=int, default=None,
                        dest="max_train_steps", help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")

    # save 
    parser.add_argument("--save-dir", type=str, default='./saved',
                        dest="save_dir", help='save directory')
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        dest="resume_from_checkpoint", help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `latest` to automatically select the last available checkpoint.")
    parser.add_argument("--checkpointing-steps", type=int, default=500,
                        dest="checkpointing_steps", help="Save a checkpoint of the training state every X updates. ")
    parser.add_argument("--save-model-epochs", type=int, default=10,
                        dest="save_model_epochs", help="Save the model every X epochs.")
    
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
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
    )
    if args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    logger.info(accelerator.state, main_process_only=False)

    # Initialize save path
    os.makedirs(args.save_dir, exist_ok=True)

    # load unet, vae, text_encoder, tokenizer, noise_scheduler
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, revision=args.revision, local_files_only=True)
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = pipe.scheduler

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ####

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Load dataset
    train_dataloader = get_train_loader(args, tokenizer) 

    # Set LoRA layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, 
                                                  cross_attention_dim=cross_attention_dim,
                                                  rank=128,
                                                 )

    unet.set_attn_processor(lora_attn_procs)
    
    ####
    for n, m in unet.named_modules():
        if ('attn' in n) and (isinstance(m, torch.nn.Dropout)):
            m.p = 0.1

    lora_layers = AttnProcsLayers(unet.attn_processors)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (accelerator.num_processes*args.gradient_accumulation_steps))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    args.lr_warmup_steps = math.ceil((num_update_steps_per_epoch * args.num_train_epochs)*0.1)
    
    if args.checkpointing_steps==-1:
        args.checkpointing_steps=math.ceil((num_update_steps_per_epoch * args.num_train_epochs)*0.01)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_name, config=vars(args))

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
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
            accelerator.load_state(os.path.join(args.save_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                # latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.mode()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")            

                ####
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.save_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.save_dir)

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
