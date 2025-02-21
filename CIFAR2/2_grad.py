import os
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F   
from torch.func import functional_call, vmap, grad

from diffusers import DDPMScheduler, UNet2DModel

from trak.projectors import ProjectionType, CudaProjector

from Tools.Dataloader.cifar2 import get_train_loader, get_test_loader, get_gen_loader

dataset_len = {
    'train': 5000,
    'test': 1000,
    'gen': 10000,
}


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def vectorize_and_ignore_buffers(g, params_dict=None):
    """
    gradients are given as a tuple :code:`(grad_w0, grad_w1, ... grad_wp)` where
    :code:`p` is the number of weight matrices. each :code:`grad_wi` has shape
    :code:`[batch_size, ...]` this function flattens :code:`g` to have shape
    :code:`[batch_size, num_params]`.
    """
    batch_size = len(g[0])
    out = []

    for b in range(batch_size):
        out.append(torch.cat([x[b].flatten() for x in g]))
    return torch.stack(out)


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
    
    # gradient and projector
    parser.add_argument("--selected-timesteps", type=int, default=10,
                        dest="selected_timesteps", help="The selected timesteps for the computation of output function")
    parser.add_argument("--selected-timesteps-strategy", type=str, default='uniform',
                        dest="selected_timesteps_strategy", choices=['uniform', 'cumulative'], 
                        help="The strategy of timestep selection")
    parser.add_argument("--output-type", type=str, default='das',
                        dest="output_type", choices=['das', 'dtrak'],
                        help="The output function type, can be das or dtrak")
    parser.add_argument("--proj-dim", type=int, default=4096, 
                        dest="proj_dim", help="The dimension of gradients after projection")
    
    # save
    parser.add_argument("--save-dir", type=str, default="./saved/grads/",
                        dest="save_dir", help="The output directory where the model gradients will be written.")

    
    args = parser.parse_args()
    return args


def main(args):
    
    # seed
    set_seeds(args.seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    dataset_type = 'train' if 'idx-train' in args.index_path else 'test' if 'idx-test' in args.index_path else 'gen' if 'idx-gen' in args.index_path else None
    get_loader = get_train_loader if dataset_type=='train' else get_test_loader if dataset_type=='test' else get_gen_loader if dataset_type=='gen' else None
    dataloader = get_loader(args)

    # initialize ddpm 
    model = get_model()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
    )
    
    # Scheduler and math around the number of training steps.
    model_path = '{}/unet'.format(args.model_dir)
    model.from_pretrained(model_path)
    model.cuda()
    model.eval()

    param_num = count_parameters(model)
    print(f"Number of parameters: {param_num//1e6:.2f}M")

    # get params and buffers
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad==True}
    buffers = {k: v.detach() for k, v in model.named_buffers() if v.requires_grad==True}

    ####

    # define output function
    def mse_output(params, buffers, noisy_latents, timesteps, targets):
        noisy_latents = noisy_latents.unsqueeze(0)
        timesteps = timesteps.unsqueeze(0)
        targets = targets.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), args=noisy_latents, 
                                kwargs={'timestep': timesteps, })
        predictions = predictions.sample

        f = F.mse_loss(predictions.float(), torch.zeros_like(targets).float(), reduction="none")

        f = f.reshape(1, -1)
        f = f.mean()
        return f
            
    def l1_output(params, buffers, noisy_latents, timesteps, targets):
        noisy_latents = noisy_latents.unsqueeze(0)
        timesteps = timesteps.unsqueeze(0)
        targets = targets.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), args=noisy_latents, 
                                kwargs={'timestep': timesteps, })
        predictions = predictions.sample
        
        f = F.l1_loss(predictions.float(), torch.zeros_like(targets).float(), reduction="none")

        f = f.reshape(1, -1)
        f = f.mean()
        return f
    if args.output_type=='das':
        output_function = l1_output
    elif args.output_type=='dtrak':
        output_function = mse_output
    grad_output = grad(output_function)

    # normalize factor
    normalize_factor = torch.sqrt(
        torch.tensor(count_parameters(model), dtype=torch.float32)
    )

    # initialize projector
    projector = CudaProjector(
        grad_dim=param_num, 
        proj_dim=args.proj_dim,
        seed=0, 
        proj_type=ProjectionType.normal,
        device=device,
        max_batch_size=32,
    )

    # initialize save np array
    ####
    filename = os.path.join(args.save_dir, f'ddpm-{args.output_type}-{dataset_type}-t{args.selected_timesteps}-d{args.proj_dim}-es{args.e_seed}.npy')
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    #  TBD: len of dataset
    dstore_keys = np.memmap(filename, 
                            dtype=np.float32, 
                            mode='w+', 
                            shape=(dataset_len[dataset_type], args.proj_dim))   
      
    ####
    index_start = 0
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
            
        # Skip steps until we reach the resumed step
        latents = batch["input"].to(device)
        bsz = len(latents)
        index_end = index_start + bsz
        ####
        if args.selected_timesteps_strategy=='uniform':
            selected_timesteps = range(0, 1000, 1000//args.selected_timesteps)
        elif args.selected_timesteps_strategy=='cumulative':
            selected_timesteps = range(0, args.selected_timesteps)            
        ####
        for index_t, t in enumerate(selected_timesteps):

            timesteps = torch.tensor([t]*bsz, device=device)
            timesteps = timesteps.long()

            # set a seed to sample noise for each timestep
            set_seeds(args.e_seed*1000+t)
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
                
            ####
            t_grads = vmap(
                grad_output, 
                in_dims=(None, None, 0, 0, 0),
            )(params, buffers, noisy_latents, timesteps, target)
            project_grads = projector.project(t_grads, model_id=0)
            normalized_grads = project_grads / normalize_factor

            if index_t==0:
                grads = normalized_grads
            else:
                grads += normalized_grads
            
        grads = grads / args.selected_timesteps
            
        #save gradients
        dstore_keys[index_start:index_end] = grads.to().cpu().clone().detach().numpy()
        index_start = index_end

if __name__ == "__main__":
    args = parse_args()
    main(args)
