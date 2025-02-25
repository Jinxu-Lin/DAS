import os
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from diffusers import DDPMScheduler, UNet2DModel

from CIFAR2.cifar2 import get_train_loader, get_test_loader, get_gen_loader

def set_seeds(seed):
    
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


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # seed
    parser.add_argument("--seed", type=int, default=42,
                        dest="seed", help="random seed")
    parser.add_argument("--e_seed", type=int, default=0, help="seed for sampling noise")

    # dataset
    parser.add_argument("--dataset", type=str, default="cifar2",
                        dest="dataset", help="dataset name")
    parser.add_argument("--load-dataset", action="store_true", default=False,
                        dest="load_dataset", help='load local dataset')
    parser.add_argument("--dataset-dir", type=str, default="../Dataset/CIFAR10",
                        dest="dataset_dir", help='dataset directory')
    parser.add_argument("--dataset-type", type=str, default='train',
                        dest="dataset_type", choices=['train', 'test', 'gen'], 
                        help='type of dataset')
    parser.add_argument("--index-path", type=str, default=None,
                        dest="index_path", help='index path of dataset')
    parser.add_argument("--gen-path", type=str, default=None,
                        dest="gen_path", help="path of generated images")
    
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
    parser.add_argument("--batch-size", type=int, default=32, 
                        dest="batch_size", help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0,
                        dest="dataloader_num_workers", help="The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")

    # ddpm
    parser.add_argument("--model-dir", type=str, default='./saved/models/',
                        dest="model_dir", help='The directory of the model')
    parser.add_argument("--prediction-type", type=str, default="epsilon",
                        dest="prediction_type", choices=["epsilon", "v_prediction"],
                        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.")
    parser.add_argument("--ddpm-num-steps", type=int, default=1000,
                        dest="ddpm_num_steps", help="number of inference steps for DDPM")
    parser.add_argument("--ddpm-beta-schedule", type=str, default="linear",
                        dest="ddpm_beta_schedule", choices=["linear", "cosine"],
                        help="beta schedule for DDPM")    
    
    # save
    parser.add_argument("--save-dir", type=str, default="./saved/errors/",
                        dest="save_dir", help="The output directory where the model errors will be written.")
    
    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def main(args):    
    
    # set seed
    set_seeds(args.seed)
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    dataset_type = 'train' if 'idx-train' in args.index_path else 'test' if 'idx-test' in args.index_path else 'gen' if 'idx-gen' in args.index_path else None
    get_loader = get_train_loader if dataset_type=='train' else get_test_loader if dataset_type=='test' else get_gen_loader if dataset_type=='gen' else None
    dataloader = get_loader(args)

    # init ddpm
    model = get_model()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
    )

    # load model
    model_path = '{}/unet'.format(args.model_dir)
    model.from_pretrained(model_path)
    model.cuda()
    model.eval()
    
    batch_loss_list = []
    for step, batch in enumerate(dataloader):
        
        print(f"{step}/{len(dataloader)}")

        latents = batch["input"].to(device)
        bsz = len(latents)
        
        ####
        time_loss_list = []
        for t in range(args.ddpm_num_steps):

            timesteps = torch.tensor([t]*bsz, device=device)
            timesteps = timesteps.long()
        
            # set a seed to sample noise for each timestep
            set_seeds(args.e_seed*1000+t) # set seeds !!!
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            # Predict the noise residual and compute loss
            with torch.no_grad():
                model_pred = model(noisy_latents, timesteps).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape))))
            time_loss_list.append(loss.detach().cpu().numpy())
        ####
        batch_loss_list.append(time_loss_list)
        
    save_dir = f'{args.save_dir}/{dataset_type}-es{args.e_seed}.pkl'
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    with open(save_dir, 'wb') as handle:
        pickle.dump(batch_loss_list, handle)


if __name__ == "__main__":
    args = parse_args()
    main(args)
