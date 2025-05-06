# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable
import os
import torch
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import PIL
import torchvision.transforms as T
import utils
import numpy as np
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from PIL import Image
from matplotlib import cm
import torch.nn as nn
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2


import torch
import matplotlib.pyplot as plt
import os

def save_reconstruction(images, outputs, targets, epoch, idx, output_dir):
    """ Saves and visualizes reconstructed images. """
    images, outputs, targets = images.cpu(), outputs.cpu(), targets.cpu()
    
    fig, axes = plt.subplots(3, len(images), figsize=(12, 6))
    for i in range(len(images)):
        axes[0, i].imshow(images[i].permute(1, 2, 0), cmap="gray")
        axes[0, i].set_title("Input")

        axes[1, i].imshow(outputs[i].permute(1, 2, 0), cmap="gray")
        axes[1, i].set_title("Reconstruction")

        axes[2, i].imshow(targets[i].permute(1, 2, 0), cmap="gray")
        axes[2, i].set_title("Ground Truth")

        for ax in axes[:, i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"reconstruction_epoch{epoch}_batch{idx}.png"))
    plt.close()


def test_one_epoch_visuals(
                        model,
                        data_loader_test,
                        optimizer,
                        device,
                        epoch,
                        args,
                        loss_scaler,
                        clip_grad,
                        log_writer=None,
                        start_steps=0,
                        lr_schedule_values=None,
                        wd_schedule_values=None,
                        patch_size=16,
                        normlize_target=False):

    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    os.makedirs(args.output_dir, exist_ok=True)  # Create output directory if not exists

    with torch.no_grad():  # No need to compute gradients during evaluation
        for idx, batch in enumerate(data_loader_test):
            # print('count: ',i)
            # i+=1
            # print('item len: ',sys.getsizeof(item))
            # print('item: ',item)
            # break
            image, targets = batch[0]
            images, targets = images.to(device), targets.to(device)  # Move to device

            # Forward pass
            outputs = model(images)

            # Compute loss (if needed)
            loss = torch.nn.functional.mse_loss(outputs, targets)  # Example loss function
            total_loss += loss.item()

            # Save or visualize reconstructions
            save_reconstruction(images, outputs, targets, epoch, idx, args.output_dir)

    avg_loss = total_loss / len(data_loader_test)  # Compute average loss

    # Log results if log_writer is provided
    if log_writer:
        log_writer.add_scalar("Test Loss", avg_loss, epoch)

    print(f"Epoch {epoch}: Test Loss = {avg_loss:.4f}")

    return {"loss": avg_loss}  # Return statistics



def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, save_dir=None, delta=2.0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss(reduction="none")

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it%len(lr_schedule_values)] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it%len(wd_schedule_values)]

        videos, bool_masked_pos, decode_masked_pos, vid_name, fname = batch
        
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            # labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            prior_mask = False
            if step>5:
                prior_mask = True
            mask_outputs, p_x, bool_masked_pos = model(videos, priors = bool_masked_pos, prior_mask= prior_mask, delta=delta)

            # mask_labels (lbls in the original order)
            mask_labels = videos_patch[bool_masked_pos].reshape(B, -1, C) # vis_labels = videos_patch[~bool_masked_pos].reshape(B, -1, C)
            # losses
            # Reconstruction loss: l_r -> B, N_m (for all tokens)
            mask_l_r = torch.mean(loss_func(input=mask_outputs, target=mask_labels), dim=-1)

            # Sampling loss: l_s -> B, N_m
            l_s =torch.zeros(videos.shape[0], ).to(mask_l_r.device)
            for i in range(p_x.shape[0]):
                # categorical distribution
                m = torch.distributions.categorical.Categorical(probs=p_x[i])
                
                # log-probabilities
                log_probs = m.log_prob(torch.arange(0, p_x.shape[1], 1).to(p_x.device)) # 1, N_m
                
                # mask log-probs
                mask_log_probs = log_probs[bool_masked_pos[i]]

                # we need to select tokens that maximize the reconstruction error, so (-) sign
                l_s[i] = -torch.mean(mask_log_probs*mask_l_r[i].detach())
                
            # Total loss
            m_l_r = torch.mean(mask_l_r) #Reconstruction loss
            m_l_s = 1e-4*torch.mean(l_s) #Sampling loss
            loss = m_l_r + m_l_s #Total loss
            # loss = m_l_r

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_reconstruction=m_l_r.item(), head="loss")
            log_writer.update(loss_sampling=m_l_s.item(), head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()
        
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_focusmae(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, save_dir=None, delta=2.0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss(reduction="none")

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it%len(lr_schedule_values)] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it%len(wd_schedule_values)]

        # videos, _,  _, vid_name = batch
        videos, bool_masked_pos, decode_masked_pos, vid_name, fname = batch
    

        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
        
        with torch.cuda.amp.autocast():
            prior_mask = False
            if step>5:
                prior_mask = True
            mask_outputs, p_x, bool_masked_pos = model(videos, priors = bool_masked_pos, prior_mask= prior_mask, delta=delta)
            # print("p_x:", p_x)
            # print("Sum of p_x:", p_x.sum())

            # mask_labels (lbls in the original order)
            mask_labels = videos_patch[bool_masked_pos].reshape(B, -1, C) # vis_labels = videos_patch[~bool_masked_pos].reshape(B, -1, C)
            
            # losses
            # Reconstruction loss: l_r -> B, N_m (for all tokens)
            mask_l_r = torch.mean(loss_func(input=mask_outputs, target=mask_labels), dim=-1)

            # Sampling loss: l_s -> B, N_m
            l_s =torch.zeros(videos.shape[0], ).to(mask_l_r.device)
            for i in range(p_x.shape[0]):
                # categorical distribution
                m = torch.distributions.categorical.Categorical(probs=p_x[i])
                
                # log-probabilities
                log_probs = m.log_prob(torch.arange(0, p_x.shape[1], 1).to(p_x.device)) # 1, N_m
                
                # mask log-probs
                mask_log_probs = log_probs[bool_masked_pos[i]]

                # we need to select tokens that maximize the reconstruction error, so (-) sign
                l_s[i] = -torch.mean(mask_log_probs*mask_l_r[i].detach())
                
            # Total loss
            m_l_r = torch.mean(mask_l_r) #Reconstruction loss
            m_l_s = 1e-4*torch.mean(l_s) #Sampling loss
            loss = m_l_r + m_l_s #Total loss
            # loss = m_l_r

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_reconstruction=m_l_r.item(), head="loss")
            # log_writer.update(loss_sampling=m_l_s.item(), head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()
        
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
