import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from ddpm_schedule import linear_beta_schedule
from ddpm_model import Unet

import os, re, subprocess, sys

timesteps = 1000

epochs = 100000
batch_size = 16
learningRate = 1e-4

# U-net parameters
dim = 32
dim_mults = [1, 2, 4]
channels = 3

lossIdxMin = 30000 # start storing model and optimizer if loss < lossIdxMin

nrHistory = 10

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i)
    return img.clamp(-1, 1)

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def main():

    global lossIdxMin

    argv = sys.argv
    argc = len(argv)

    print('%s trains ddpm model' % argv[0])
    print('[usage] python %s <training data(.npy)> [<model to be restarterd(.pth)> <optimizer to be restarted(.pth)>]' % argv[0])

    if argc < 2:
        quit()

    torch.manual_seed(42)

    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda t: (t * 2) - 1)
    ])
   
    images = np.load(argv[1])

    imgs = []
    for image in images:
        imgs.append(transform(image))

    nrData = len(imgs)
    nrBatches = nrData // batch_size
    if nrData % batch_size != 0:
        nrBatches += 1

    dl = DataLoader(imgs, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(dim=dim, dim_mults=dim_mults, channels=channels).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learningRate)

    if argc > 2:
        model.load_state_dict(torch.load(argv[2], map_location=device))

        base = os.path.basename(argv[2])
        filename = os.path.splitext(base)[0]
        lossIdxMin = int(re.sub(r'.*_', '', filename))

    if argc > 3:
        optimizer.load_state_dict(torch.load(argv[3], map_location=device))

    def handle_batch(batch):
        batch_size = batch.shape[0]
        batch = batch.to(device)

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        loss = p_losses(model, batch, t, loss_type='l2')
        return loss

    models = []
    optimizers = []

    for epoch in range(epochs):
        losses = list()
        
        for i, batch in enumerate(dl):

            print('epoch:%d/%d batch:%d/%d' % (epoch+1, epochs, i+1, nrBatches))

            optimizer.zero_grad()
            loss = handle_batch(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        train_loss = np.mean(losses)
        lossIdx = int(train_loss * 100000)
        print(lossIdx)

        if lossIdx < lossIdxMin:
            filename_model = 'model_%d.pth' % lossIdx
            torch.save(model.state_dict(), filename_model)
            models.append(filename_model)

            filename_optimizer = 'optimizer_%d.pth' % lossIdx
            torch.save(optimizer.state_dict(), filename_optimizer)
            optimizers.append(filename_optimizer)

            while len(models) > nrHistory:
                cmd = 'del %s' % filename_model
                subprocess.run(cmd, shell=True)
                del(models[0])
                cmd = 'del %s' % filename_optimizer
                subprocess.run(cmd, shell=True)
                del(optimizers[0])

            lossIdxMin = lossIdx

if __name__ == '__main__':
    main()
