import cv2, os, sys
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

nrRows = 2
nrCols = 2

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

# hyperparameters
input_dim = SIZE * SIZE * 3
hidden_dim = 50
latent_dim = 10

z2 = None

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, use_sigmoid=False):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.use_sigmoid = use_sigmoid

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        if self.use_sigmoid:
            h = torch.sigmoid(h)
        return h

def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    z = mu
    return z


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder1 = Encoder(input_dim, hidden_dim, latent_dim)
        self.encoder2 = Encoder(latent_dim, hidden_dim, latent_dim)
        self.decoder1 = Decoder(latent_dim, hidden_dim, input_dim, use_sigmoid=True)
        self.decoder2 = Decoder(latent_dim, hidden_dim, latent_dim)

    def get_loss(self, x):
        mu1, sigma1 = self.encoder1(x)
        z1 = reparameterize(mu1, sigma1)
        mu2, sigma2 = self.encoder2(z1)
        z2 = reparameterize(mu2, sigma2)

        z_hat = self.decoder2(z2)
        x_hat = self.decoder1(z1)

        # loss
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction='sum')
        L2 = - torch.sum(1 + torch.log(sigma2 ** 2) - mu2 ** 2 - sigma2 ** 2)
        L3 = - torch.sum(1 + torch.log(sigma1 ** 2) - (mu1 - z_hat) ** 2 - sigma1 ** 2)
        return (L1 + L2 + L3) / batch_size

    def encode(self, x):

        mu1, sigma1 = self.encoder1(x)
        y = reparameterize(mu1, sigma1)
        mu2, sigma2 = self.encoder2(y)

        return mu2, sigma2

def img2seed(image):

    image = cv2.resize(image, (SIZE, SIZE))
    image = image.astype(np.float32) / 255.0
    image = image.reshape((1, SIZE * SIZE * 3)) 

    img = torch.from_numpy(image).clone()

    mu, sigma = model.encode(img)
    seed = reparameterize(mu, sigma)

    return seed

def generate_images(model): 

    global z2

    screen = np.empty((SIZE * nrRows, SIZE * nrCols, 3), np.float32)
    
    z2[0] = torch.randn(1, latent_dim)
   
    seed1 = torch.randn(1, latent_dim)
    seed2 = torch.randn(1, latent_dim)
    seed3 = torch.randn(1, latent_dim)
    
    # visualize generated images
    with torch.no_grad():

        for r in range(nrRows):
            r2 = nrRows - 1 - r
            z2[r * nrCols] = (z2[0] * r2 + seed1 * r) / (nrRows - 1)
            z2[r * nrCols + nrCols - 1] = (seed2 * r2 + seed3 * r) / (nrRows - 1)

        for r in range(nrRows):
            left = z2[r * nrCols]
            right = z2[r * nrCols + nrCols - 1]

            for c in range(1, nrCols - 1):
                c2 = nrCols - 1 - c
                z2[r * nrCols + c] = (left * c2 + right * c) / (nrCols - 1)

        z2 = torch.randn(nrRows * nrCols, latent_dim)

        z1_hat = model.decoder2(z2)
        z1 = reparameterize(z1_hat, torch.ones_like(z1_hat))
        x = model.decoder1(z1)
        generated_images = x.view(nrRows * nrCols, 3, SIZE, SIZE)

    for R in range(nrRows):
        y = R * SIZE
        for C in range(nrCols):
            x = C * SIZE
            img = generated_images[R * nrCols + C]
            #img = np.transpose(img,(0,1,2))
            img = img.reshape((SIZE * SIZE * 3))
            img = img.reshape((SIZE, SIZE, 3))
            screen[y:y+SIZE,x:x+SIZE,:] = img

    return screen, generated_images

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

    screen = np.zeros((nrRows * SIZE, nrCols * SIZE, 3), np.uint8)

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i)

        imgs = img.to('cpu').detach().numpy()
        imgs = np.transpose(imgs, (0, 2, 3, 1)) # n,c,h,w --> n,h,w,c
        imgs += 1   # -1~1 --> 0~2
        imgs /= 2   # 0~1
        imgs *= 255 # 0~255
        imgs = np.clip(imgs, 0, 255)
        imgs = imgs.astype(np.uint8)

        for r in range(nrRows):
            top = r * SIZE
            bottom = top + SIZE
            
            for c in range(nrCols):
                left = c * SIZE
                right = left + SIZE

                idx = r * nrCols + c

                screen[top:bottom, left:right] = imgs[idx]

        cv2.imwrite('%04d.png' % (i+1), screen)

    return img.clamp(-1, 1)

@torch.no_grad()
def sample(model, seeds):
    return p_sample_loop(model, seeds)

def draw_cursor(screen, r, c):

    left = c * SIZE
    right = left + SIZE
    top = r * SIZE
    bottom = top + SIZE
         
    dst = screen.copy()
    dst = cv2.rectangle(dst, (left, top), (right, bottom), (255, 0, 255), 2)

    return dst

def main():

    global nrRows, nrCols, z2

    argv = sys.argv
    argc = len(argv)

    print('%s generates image using DDPM' % argv[0])
    print('[usage] python %s <VAE params> <DDPM params> [<nrRows> <nrCols>]' % argv[0])
    
    paramsVAE = os.path.join(os.path.dirname(__file__), '..\\data\\vae_1120.pth')
    if argc > 1:
        paramsVAE = argv[1]

    paramsDDPM = os.path.join(os.path.dirname(__file__), '..\\data\\ddpm_model.pth')
    if argc > 2:
        paramsDDPM = argv[2]

    if argc > 3:
        nrRows = int(argv[3])

    if argc > 4:
        nrCols = int(argv[4])

    torch.manual_seed(42)

    z2 = torch.randn(nrRows * nrCols, latent_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    modelVAE = VAE(input_dim, hidden_dim, latent_dim)
    modelVAE.load_state_dict(torch.load(paramsVAE, map_location=device))

    modelDDPM = Unet(dim=32, dim_mults=[1, 2, 4], channels=3).to(device)
    modelDDPM.load_state_dict(torch.load(paramsDDPM, map_location=device))
   
    shape = [nrRows * nrCols, nrChannels, SIZE, SIZE] 

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

        imgsVAE, seeds = generate_images(modelVAE)
        cv2.imwrite('VAE.png', imgsVAE * 255)
        print('save VEA.png')

        seeds = seeds * SCALE - SCALE / 2
        seeds = seeds.numpy()
        

        seeds = seeds.reshape((nrRows * SIZE * nrCols * SIZE * 3))
        seeds = seeds.reshape((nrRows * nrCols, SIZE, SIZE, 3))
        seeds = np.transpose(seeds, (0, 3, 1, 2))
        seeds = torch.tensor(seeds)

        imgs = sample(modelDDPM, seeds.to(device))
        imgs = imgs.to('cpu').detach().numpy()

        imgs = np.transpose(imgs, (0, 2, 3, 1)) # n,c,h,w --> n,h,w,c
        imgs += 1   # -1~1 --> 0~2
        imgs /= 2   # 0~1
        imgs *= 255 # 0~255
        imgs = imgs.astype(np.uint8)

        screen = np.zeros((nrRows * SIZE, nrCols * SIZE, 3), np.uint8)

        for r in range(nrRows):
            top = r * SIZE
            bottom = top + SIZE
            
            for c in range(nrCols):
                left = c * SIZE
                right = left + SIZE

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
                top = r * SIZE
                bottom = top + SIZE
                left = c * SIZE
                right = left + SIZE

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
            cv2.imshow('DDPM', dst) 
            cv2.imshow('VAE', imgsVAE)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
