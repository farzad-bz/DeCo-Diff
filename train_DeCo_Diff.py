import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from collections import OrderedDict
import json
from time import time
from PIL import Image
from copy import deepcopy
from glob import glob
import argparse
import logging
import os
import torch.nn.functional as F
from models import UNET_models

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from MVTECDataLoader import MVTECDataset
from VISADataLoader import VISADataset
from scipy.ndimage import gaussian_filter
from transformers import get_cosine_schedule_with_warmup


import torch.nn as nn
class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=15):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, c):
        c = self.embedding(c)
        return c


def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

# @torch.no_grad()
# def update_ema(ema_model, model, decay=0.9999):
#     """
#     Step the EMA model towards the current model.
#     """
#     ema_params = OrderedDict(ema_model.named_parameters())
#     model_params = OrderedDict(model.named_parameters())

#     for name, param in model_params.items():
#         ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def shuffle_patches(image, patch_size):
    N, C, H, W = image.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "Image dimensions should be divisible by patch size."

    # Extract patches
    unfolded = F.unfold(image, kernel_size=patch_size, stride=patch_size)  # Shape: (N*C*P*P, num_patches)

    # Reshape unfolded patches to (N, C, P, P, num_patches)
    num_patches = unfolded.shape[-1]
    unfolded = unfolded.view(N, C, P, P, num_patches)

    # Shuffle patches across the batch dimension
    unfolded = unfolded.permute(0, 4, 1, 2, 3)  # Shape: (N, num_patches, C, P, P)
    unfolded = unfolded.reshape(N * num_patches, C, P, P)  # Shape: (N * num_patches, C, P, P)

    # Shuffle patches
    indices = torch.randperm(N * num_patches)
    shuffled_unfolded = unfolded[indices]

    # Reshape back to original format
    shuffled_unfolded = shuffled_unfolded.view(N, num_patches, C, P, P)
    shuffled_unfolded = shuffled_unfolded.permute(0, 2, 3, 4, 1)  # Shape: (N, C, P, P, num_patches)

    # Reconstruct the image
    shuffled_unfolded = shuffled_unfolded.contiguous().view(N * C * P * P, num_patches)
    folded = F.fold(shuffled_unfolded, output_size=(H, W), kernel_size=patch_size, stride=patch_size)

    # Fold operation does not include channels; need to reshape and combine
    folded = folded.view(N, C, H, W)
    
    return folded



def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def random_mask(x : torch.Tensor, mask_ratios, mask_patch_size=1):
    for mask_ratio in mask_ratios:
        assert mask_ratio >=0 and mask_ratio<=1
    n, c, w, h = x.shape
    size = int(np.prod(x.shape[2:]) / (mask_patch_size**2))
    mask = torch.zeros((n,c,size)).to(x.device)
    for b in range(n):
        masked_indexes = np.arange(size)
        np.random.shuffle(masked_indexes)
        masked_indexes = masked_indexes[:int(size * (1 - mask_ratios[b]))]
        mask[b,:, masked_indexes] = 1
    mask = mask.reshape(n, c, int(w/mask_patch_size), int(w/mask_patch_size))
    mask = mask.repeat_interleave(mask_patch_size, dim=2).repeat_interleave(mask_patch_size, dim=3)
    return mask


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        
        with open(f'{args.results_dir}/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.model_size.replace('/', '-')}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.center_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.actual_image_size // 8
    model = UNET_models[args.model_size](latent_size=latent_size)
        

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="ddim10", predict_deviation=True, predict_xstart=False, sigma_small=False)  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}").to(device)
    vae.eval()
    logger.info(f"Number of Parameters: {sum(p.numel() for p in model.parameters()):}")


    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
        
    
    if args.dataset=='mvtec':
        dataset = MVTECDataset('train', object_class=args.object_category, rootdir=args.data_dir, transform=transform, image_size=args.image_size,  center_size=args.center_size, augment=args.augmentation, center_crop=args.center_crop)
    elif args.dataset=='visa':
        dataset = VISADataset('train', object_class=args.object_category, rootdir=args.data_dir, transform=transform, image_size=args.image_size,  center_size=args.center_size, augment=args.augmentation, center_crop=args.center_crop)
       
    batch_size = args.global_batch_size // dist.get_world_size()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    accumulation_steps = 1

        
    logger.info(f"Dataset contains {len(dataset):,} training images")

    adjusted_epochs = args.epochs

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=args.warmup_epochs,
        num_training_steps=args.epochs*1.5,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adjusted_epochs, eta_min=args.lr/100)
    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
        
    # Prepare models for training:
    # update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    # ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_mse = 0
    start_time = time()

    logger.info(f"Training for {adjusted_epochs} epochs...")
    
    for epoch in range(adjusted_epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for ii, (x, _, y) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
 
            if args.actual_image_size == 224:
                mask_patch_size = np.random.choice([1,2,4,7], 1, p=[0.4, 0.3, 0.2, 0.1]).item()
            if args.actual_image_size == 256:
                mask_patch_size = np.random.choice([1,2,4,8], 1, p=[0.4, 0.3, 0.2, 0.1]).item()
            if args.actual_image_size == 320:
                mask_patch_size = np.random.choice([1,2,4,8], 1, p=[0.4, 0.3, 0.2, 0.1]).item()
            if args.actual_image_size == 384:
                mask_patch_size = np.random.choice([1,2,4,8,12], 1, p=[0.3, 0.25, 0.20, 0.15, 0.1]).item()
            if args.actual_image_size == 448:
                mask_patch_size = np.random.choice([1,2,4,8,14], 1, p=[0.3, 0.25, 0.20, 0.15, 0.1]).item()
            elif args.actual_image_size == 512:
                mask_patch_size = np.random.choice([1,2,4,8,16], 1, p=[0.3, 0.25, 0.20, 0.15, 0.1]).item()   
            if args.mask_random_ratio:
                mask_ratios = np.random.uniform(low=0.0, high=0.7, size = x.shape[0])
            else:
                mask_ratio = args.mask_ratio
                mask_ratios = [mask_ratio]*x.shape[0],
                
            mask = random_mask(x, mask_ratios=mask_ratios, mask_patch_size=mask_patch_size)
    
            model_kwargs = {
            'context' : torch.tensor(y).to(device).int().unsqueeze(1),
            'mask': mask
            }
            
            noise_mask = random_mask(x, mask_ratios=np.random.uniform(low=0.0, high=args.patch_shuffle_ratio, size = x.shape[0]), mask_patch_size=mask_patch_size)
            noise = noise_mask * torch.randn_like(x, device=device) + (1-noise_mask) *  shuffle_patches(x, mask_patch_size)
            
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs, noise = noise)
            loss = loss_dict["loss"].mean()
            loss.backward()
            
            if (ii + 1) % accumulation_steps == 0:
                opt.step()
                opt.zero_grad() 
                
            # update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            running_mse += loss_dict["mse"].mean().item()
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_mse = torch.tensor(running_mse / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_mse, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_mse = avg_mse.item() / dist.get_world_size()
                logger.info(f"(category={args.object_category} step={train_steps:07d}) MSE Loss: {avg_mse:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                running_mse = 0
                running_mt = 0
                log_steps = 0
                start_time = time()

        scheduler.step()
        if epoch % args.ckpt_every == 0 and epoch>0:
            if rank == 0: 
                # Save checkpoint:
                checkpoint = {
                    "model": model.module.state_dict(),
                    # "ema": ema.state_dict(),
                    # "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/last.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            dist.barrier()
            

    logger.info("Done!")
    cleanup()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['mvtec','visa'], default="mvtec")
    parser.add_argument("--data-dir", type=str, default='./mvtec-dataset/')
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_XS')
    parser.add_argument("--image-size", type=int, default= 288 )
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=10)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.7)
    parser.add_argument("--patch-shuffle-ratio", type=float, default=0.3)
    parser.add_argument("--object-category", type=str, default='all')
    parser.add_argument("--mask-random-ratio", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--from-scratch", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--augmentation", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    
    
    args = parser.parse_args()
    if args.dataset == 'mvtec':
        args.num_classes = 15
    elif args.dataset == 'visa':
        args.num_classes = 12
    args.results_dir = f"./DeCo-Diff_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
    if args.center_crop:
        args.results_dir += "_CenterCrop"
        args.actual_image_size = args.center_size
    else:
        args.actual_image_size = args.image_size
        
    main(args)
