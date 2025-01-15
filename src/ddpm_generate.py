import cv2, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import make_grid

from ddpm_schedule import linear_beta_schedule
from ddpm_model import Unet
from tqdm import tqdm

imageSize = 128
nrChannels = 3

nrRows = 5
nrCols = 8

timesteps = 1000

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

def draw_cursor(screen, r, c):

    left = c * imageSize
    right = left + imageSize
    top = r * imageSize
    bottom = top + imageSize
         
    dst = screen.copy()
    dst = cv2.rectangle(dst, (left, top), (right, bottom), (255, 0, 255), 2)

    return dst

def main():

    global nrRows, nrCols

    argv = sys.argv
    argc = len(argv)

    print('%s generates image using DDPM' % argv[0])
    print('[usage] python %s <DDPM model> [<nrRows> <nrCols>]' % argv[0])
    
    if argc < 2:
        quit()

    model_params = argv[1]

    if argc > 2:
        nrRows = int(argv[2])

    if argc > 3:
        nrCols = int(argv[3])

    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    #model = Unet(**config.model).to(device)
    model = Unet(dim=32, dim_mults=[1, 2, 4], channels=3).to(device)
    model.load_state_dict(torch.load(model_params, map_location=device))
   
    shape = [nrRows * nrCols, nrChannels, imageSize, imageSize] 

    fTerminate = False

    no = 1
    scrNo = 1

    print('')
    print('Hit ESC-key to terminate')
    print('Hit Arrow-key to move cursor')
    print('Hit s-key to save selected image')
    print('Hit S-key to save screen image')
    print('Hit r-key to generate new images')
    print('')
    
    while not fTerminate:

        seeds = torch.randn(shape, device=device)

        imgs = sample(model, seeds)
        imgs = imgs.to('cpu').detach().numpy()

        imgs = np.transpose(imgs, (0, 2, 3, 1)) # n,c,h,w --> n,h,w,c
        imgs += 1   # -1~1 --> 0~2
        imgs /= 2   # 0~1
        imgs *= 255 # 0~255
        imgs = imgs.astype(np.uint8)

        screen = np.zeros((nrRows * imageSize, nrCols * imageSize, 3), np.uint8)

        for r in range(nrRows):
            top = r * imageSize
            bottom = top + imageSize
            
            for c in range(nrCols):
                left = c * imageSize
                right = left + imageSize

                idx = r * nrCols + c

                screen[top:bottom, left:right] = imgs[idx]

        r = 0
        c = 0
       
        dst = draw_cursor(screen, r, c)
           
        cv2.imshow('screen', dst) 

        fSelect = True

        while fSelect:

            key = cv2.waitKeyEx(0)
        
            if key == UP:
                r -= 1
                if r < 0:
                    r = nrRows - 1
        
            elif key == DOWN:
                r += 1
                if r >= nrRows:
                    r = 0
        
            elif key == LEFT:
                c -= 1
                if c < 0:
                    c = nrCols - 1
        
            elif key == RIGHT:
                c += 1
                if c >= nrCols:
                    c = 0
        
            elif key == ord('s'):
                top = r * imageSize
                bottom = top + imageSize
                left = c * imageSize
                right = left + imageSize

                selected = screen[top:bottom, left:right]
                dst_path = '%04d.png' % no
        
                while os.path.isfile(dst_path):
                    no += 1
                    dst_path = '%04d.png' % no
        
                cv2.imwrite(dst_path, selected)
                print('save %s' % dst_path)
          
                #idx = r * nrCols + c
                #selectedSeed = seeds[idx].to('cpu').detach().numpy()
                #dst_path = '%04d.npy' % no

                #np.save(dst_path, selectedSeed)
                #print('save %s' % dst_path)

            elif key == ord('S'):
        
                dst_path = 'screen_%04d.png' % scrNo
        
                while os.path.isfile(dst_path):
                    scrNo += 1
                    dst_path = 'screen_%04d.png' % scrNo
        
                cv2.imwrite(dst_path, screen)
                print('save %s' % dst_path)

                #dst_path = 'screen_%04d.npy' % scrNo
                #np.save(dst_path, seeds.to('cpu').detach().numpy())
                #print('save %s' % dst_path)

            elif key == ord('r') or key == ord('R'):
       
                fSelect = False
                break
        
            elif key == ESC:
                fSelect = False
                fTerminate = True
                break
       
            dst = draw_cursor(screen, r, c)
            cv2.imshow('screen', dst) 

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
