import cv2, glob, os, sys
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import make_grid

from ddpm_schedule import linear_beta_schedule
from ddpm_model import Unet
from tqdm import tqdm

SIZE = 128
nrChannels = 3

SCALE = 0.5

timesteps = 300

ESC = 27
LEFT  = 2424832
UP    = 2490368
RIGHT = 2555904
DOWN  = 2621440

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
def p_sample_loop(model, seeds):
    device = next(model.parameters()).device

    b = seeds.shape[0]

    img = seeds.clone() 

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i)
        
    return img.clamp(-1, 1)

@torch.no_grad()
def sample(model, seeds):
    return p_sample_loop(model, seeds)

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s derives image from input image using DDPM model' % argv[0])
    print('[usage] python %s <wildcard for images> [<DDPM params>]' % argv[0])
   
    if argc < 2:
        quit()

    params = os.path.join(os.path.dirname(__file__), '..\\data\\ddpm_model_faces.pth')

    paths = glob.glob(argv[1])
    nrData = len(paths)

    if argc > 2:
        params = argv[2]

    output_folder = 'derived'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    model = Unet(dim=32, dim_mults=[1, 2, 4], channels=3).to(device)
    model.load_state_dict(torch.load(params, map_location=device))
   
    shape = [1, nrChannels, SIZE, SIZE] 

    for i, path in enumerate(paths):

        print('processing %d/%d: %s' % ((i+1), nrData, path))

        img = cv2.imread(path)

        cv2.imshow('source', img)
        key = cv2.waitKey(10)
        if key == 27:
            break

        H, W = img.shape[:2]
        size = np.max((H, W))
        tmp = np.zeros((size, size, 3), np.uint8)
        left = (size - W) // 2
        right = left + W
        top = (size - H) // 2
        bottom = top + H
        tmp[top:bottom, left:right, :] = img
        seed = cv2.resize(tmp, (SIZE, SIZE))
        seed = seed.astype(np.float32) / 255.0

        seed = seed * SCALE - SCALE / 2
        seed = seed[np.newaxis, :, :, :]
        seed = np.transpose(seed, (0, 3, 1, 2))
        seed = torch.tensor(seed)

        img = sample(model, seed.to(device))
        img = img.to('cpu').detach().numpy()

        img = np.transpose(img, (0, 2, 3, 1)) # n,c,h,w --> n,h,w,c
        img = np.squeeze(img) # n, h, w, c --> h, w, c
        img += 1   # -1~1 --> 0~2
        img /= 2   # 0~1
        img *= 255 # 0~255
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        base = os.path.basename(path)
        filename = os.path.splitext(base)[0]
        dst_path = os.path.join(output_folder, '%s.png' % filename)
        cv2.imwrite(dst_path, img)
        print('save %s' % dst_path)
        cv2.imshow('derived', img)

if __name__ == '__main__':
    main()
