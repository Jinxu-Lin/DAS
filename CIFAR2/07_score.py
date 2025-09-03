import os
import torch
import random
import numpy as np
import pickle
import argparse
import csv
from scipy.stats import spearmanr
import logging


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def load_error(args, logger):
    
    with open('./saved/error/ddpm-train-errors-{}.pkl'.format(args.error_num_timesteps), 'rb')  as handle:
        error_train = pickle.load(handle)
   
    error_train = np.concatenate(error_train, axis=-1)
    error_train = np.swapaxes(error_train, 0, 1)
    error_train = np.sqrt(error_train)
    
    # mean
    error_train_norm = np.linalg.norm(error_train, axis=-1, keepdims=True)
    error_train = error_train / (error_train_norm + 1e-8)
    error_train = error_train.mean(axis=-1)
    error_train = torch.from_numpy(error_train)

    logger.info(f"Successfully loaded error of shape {error_train.shape}")
    return error_train


def move_to_cpu_and_free_gpu(tensor):
    """Move tensor to CPU and free GPU memory"""
    
    if tensor.is_cuda:
        cpu_tensor = tensor.cpu()
        del tensor
        torch.cuda.empty_cache()
        return cpu_tensor
    return tensor


def get_A_B(A, B, batch_size=5000):
    """ Compute the matrix product of A and B in a batched manner. """
    
    A_gpu = A.cuda()
    B_gpu = B.cuda()

    blocks = torch.split(A_gpu, split_size_or_sections=batch_size, dim=0)
    result = torch.empty(
        (A_gpu.shape[0], B_gpu.shape[1]), dtype=A_gpu.dtype, device=A_gpu.device
    )

    for i, block in enumerate(blocks):
        start = i * batch_size
        end = min(A_gpu.shape[0], (i + 1) * batch_size)
        result[start:end] = block @ B_gpu

    del A_gpu, B_gpu
    torch.cuda.empty_cache()
    
    return result


def get_A_inv(A, lambda_reg):
    """ Compute the inverse of A. """

    A_gpu = A.cuda()
    A_reg = A_gpu + lambda_reg * torch.eye(
        A_gpu.size(0), dtype=A_gpu.dtype, device=A_gpu.device
    )
    A_inv = torch.linalg.inv(A_reg)

    A_inv /= A_inv.abs().mean()

    del A_gpu, A_reg
    torch.cuda.empty_cache()

    return A_inv


def parse_args():
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # Parameters for score computation
    parser.add_argument("--dataset_type", type=str, default='val', help="val or gen")
    parser.add_argument("--method", type=str, default='dtrak', help="dtrak or ours")
    
    # Parameters for gradient computation
    parser.add_argument("--grad_path", type=str, default='./saved/grad')
    parser.add_argument("--loss_function_type", type=str, default='mean', help="loss function type for gradient computation")
    parser.add_argument("--timestep_strategy", type=str, required=True, help="Timestep sampling strategy (uniform, cumulative)")
    parser.add_argument("--grad_num_timesteps", type=int, default=10, help="10, 100 or 1000, timesteps for gradient computation")
    parser.add_argument("--projection_dim", type=int, default=4096)
    
    # Parameters for error computation
    parser.add_argument("--error_path", type=str, default='./saved/error')
    parser.add_argument("--error_num_timesteps", type=int, default=1000, help="100, 1000 or 10000, timesteps for error computation")
    
    # Save Path
    parser.add_argument("--save_path", type=str, default='./saved/score')
    
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f"Arguments: {args}")
        
    # load model output
    loss_array_list = []
    for i in range(64):
        for seed in [0,1,2]:
            for e_seed in [0,1,2]:
                with open('./saved/lds-val/ddpm-sub-{}-{}/ddpm-{}-errors-{}.pkl'.format(i, seed, args.dataset_type, args.error_num_timesteps), 'rb')  as handle:
                    loss_list = pickle.load(handle)
                margins = np.concatenate(loss_list, axis=-1) # -logp
                ####
                if (seed==0) and (e_seed)==0:
                    loss_array = margins
                else:
                    loss_array += margins
        loss_array = loss_array/(3*3)
        loss_array_list.append(loss_array)
    lds_loss_array = np.stack(loss_array_list)
    logger.info("Successfully loaded lds_loss_array")
    logger.info(f"lds_loss_array.shape: {lds_loss_array.shape}")
    
    with open('./data/indices/idx-train.pkl', 'rb')  as handle:
        idx_train = pickle.load(handle)
    
    mask_array_list = []
    for i in range(64):
        with open('./data/indices/lds-val/sub-idx-{}.pkl'.format(i), 'rb')  as handle:
            sub_idx_train = pickle.load(handle)
        mask_array = np.in1d(idx_train, sub_idx_train)
        mask_array_list.append(mask_array)
    lds_mask_array = np.stack(mask_array_list)
    logger.info("Successfully loaded lds_mask_array")
    logger.info(f"lds_mask_array.shape: {lds_mask_array.shape}")
    
    lds_testset_correctness = lds_loss_array.mean(axis=1)
    logger.info(f"lds_testset_correctness.shape: {lds_testset_correctness.shape}")
    
    # Load gradient
    dstore_keys_list = []
    for split in range(5):
        dstore_keys = np.memmap('./saved/grad/ddpm-train-keys-{}-{}-{}-{}-{}.npy'.format(
            split, args.loss_function_type, args.grad_num_timesteps, args.projection_dim, args.timestep_strategy
            ), 
            dtype=np.float32, 
            mode='r',
            shape=(1000, args.projection_dim)
        )
        dstore_keys_list.append(dstore_keys)
    dstore_keys = np.vstack(dstore_keys_list)
    dstore_keys = torch.from_numpy(dstore_keys).cuda()
    logger.info("Successfully loaded training set gradient")
    logger.info(f"training set gradient shape: {dstore_keys.shape}")  
    
    gen_dstore_keys = np.memmap('./saved/grad/ddpm-{}-keys-0-{}-{}-{}-{}.npy'.format(
                args.dataset_type, args.loss_function_type, args.grad_num_timesteps, args.projection_dim, args.timestep_strategy
                ), 
                dtype=np.float32, 
                mode='r',
                shape=(1000, args.projection_dim)
    )
    gen_dstore_keys = torch.from_numpy(gen_dstore_keys).cuda()
    logger.info("Successfully loaded generated set gradient")
    logger.info(f"{args.dataset_type} set gradient shape: {gen_dstore_keys.shape}")    
    
    # Load train error
    error_train = load_error(args, logger)
    
    # Calculate the score
    lds_list = []
    lamb_list = [
        1e-2,2e-2,5e-2,
        1e-1,2e-1,5e-1,
        1e0, 2e0, 5e0,
        1e1, 2e1, 5e1,
        1e2, 2e2, 5e2,
        1e3, 2e3, 5e3, 
        1e4, 2e4, 5e4,  
        1e5, 2e5, 5e5, 
        1e6, 2e6, 5e6, 
    ]
    rs_list = []
    ps_list = []
    best_scores = None
    best_lds = -np.inf
    
    kernel = get_A_B(dstore_keys.T, dstore_keys)
    kernel = move_to_cpu_and_free_gpu(kernel)
    

    for lamb in lamb_list:
        
        kernel_inv =  get_A_inv(kernel, lamb)
        kernel_inv = move_to_cpu_and_free_gpu(kernel_inv)

        # compute score
        if args.method == 'dtrak':
            features = get_A_B(dstore_keys, kernel_inv)
            features = move_to_cpu_and_free_gpu(features)
            scores = get_A_B(gen_dstore_keys, features.T)
            scores = move_to_cpu_and_free_gpu(scores)
            
        elif args.method == 'DAS1':
            features = get_A_B(dstore_keys, kernel_inv)
            features = move_to_cpu_and_free_gpu(features)
            scores = get_A_B(gen_dstore_keys, features.T)
            scores = move_to_cpu_and_free_gpu(scores)
            scores = scores * error_train.unsqueeze(1)
            
        elif args.method == 'DAS1squ':
            features = get_A_B(dstore_keys, kernel_inv)
            features = move_to_cpu_and_free_gpu(features)
            scores = get_A_B(gen_dstore_keys, features.T)
            scores = move_to_cpu_and_free_gpu(scores)
            scores = scores * error_train.unsqueeze(1)
            scores = scores**2

        elif args.method == 'DAS0':
            
            features = get_A_B(dstore_keys, kernel_inv)
            features = move_to_cpu_and_free_gpu(features)
            hat_matrix = get_A_B(gen_dstore_keys, features.T)
            hat_matrix = move_to_cpu_and_free_gpu(hat_matrix)
            
            hat_value = torch.diag(hat_matrix)
            results = torch.empty(5000, args.dim, device='cuda')
            
            hat = get_A_B(kernel_inv, features.T)
            hat = move_to_cpu_and_free_gpu(hat)  
            hat = hat.T / (1 - hat_value.unsqueeze(1))**2  
            
            results = hat * error_train.unsqueeze(1) 
            
            scores = get_A_B(gen_dstore_keys, results.T)
            scores = move_to_cpu_and_free_gpu(scores)
            
        elif args.method == 'DAS0squ':
            
            features = get_A_B(dstore_keys, kernel_inv)
            features = move_to_cpu_and_free_gpu(features)
            hat_matrix = get_A_B(gen_dstore_keys, features.T)
            hat_matrix = move_to_cpu_and_free_gpu(hat_matrix)
            
            hat_value = torch.diag(hat_matrix)
            results = torch.empty(5000, args.dim, device='cuda')
            
            hat = get_A_B(kernel_inv, features.T)
            hat = move_to_cpu_and_free_gpu(hat)  
            hat = hat.T / (1 - hat_value.unsqueeze(1))**2  
            
            results = hat * error_train.unsqueeze(1) 
            
            scores = get_A_B(gen_dstore_keys, results.T)
            scores = move_to_cpu_and_free_gpu(scores)
            
            scores = scores**2 

        scores = scores.numpy()
        
        margins = lds_testset_correctness
        infl_est_ = -scores
        preds = lds_mask_array @ infl_est_.T
        logger.info(f"preds.shape: {preds.shape}")
        logger.info(f"margins.shape: {margins.shape}")
        
        # compute lds
        rs = []
        ps = []
        for ind in range(1000):
           r, p = spearmanr(preds[:, ind], margins[:, ind])
           # r, p = pearsonr(preds[:, ind], margins[:, ind])
           rs.append(r)
           ps.append(p)
        rs, ps = np.array(rs), np.array(ps)
        print(f'Correlation: {rs.mean():.3f} (avg p value {ps.mean():.6f})')
        
        rs_list.append(rs.mean())   
        ps_list.append(ps.mean())

        if rs.mean()>best_lds:
            best_scores = scores
            best_lds = rs.mean()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    with open(os.path.join(args.save_path, 'lamb_score.csv'), 'w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['lamb', 'rs'])
        for lamb, rs in zip(lamb_list, rs_list):
            writer.writerow([lamb, rs])
    
    file_path = '{}/{}/{}/{}'.format(
        args.save_path, args.dataset_type, args.projection_dim, args.grad_num_timesteps, 
    )
    with open(os.path.join(file_path, '{}_{}.pkl'.format(args.loss_function_type, best_lds)), 'wb') as handle:
        pickle.dump(args.list, handle)
            
    lds_list.append((rs_list, ps_list))
    lds_array = np.array(lds_list)
    logger.info(f"lds_array.shape: {lds_array.shape}")
    

if __name__ == "__main__":
    main()