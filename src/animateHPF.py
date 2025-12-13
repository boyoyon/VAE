import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import cv2, os, sys
import numpy as np

SIZE = 128
nrCols = 30

WAIT_TIME = 50
SCALE = 3

HPF_SCALE = 20
HPF_SCALE_MIN = 0
HPF_SCALE_MAX = 128

# hyperparameters
input_dim = SIZE * SIZE * 3
hidden_dim = 50
latent_dim = 10

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

def generate_images(model1, model2, seed0, seed1): 

    img_seq = np.empty((SIZE, SIZE * nrCols, 3), np.float32)

    z2 = torch.empty(nrCols, latent_dim)
    
    # visualize generated images
    with torch.no_grad():

        for c in range(1, nrCols - 1):
            c2 = nrCols - 1 - c
            z2[c] = (seed0 * c2 + seed1 * c) / (nrCols - 1)

        z1_hat = model1.decoder2(z2)
        z1 = reparameterize(z1_hat, torch.ones_like(z1_hat))
        x1 = model1.decoder1(z1)
        generated_images1 = x1.view(nrCols, 3, SIZE, SIZE)

        z2_hat = model2.decoder2(z2)
        z = reparameterize(z2_hat, torch.ones_like(z1_hat))
        x2 = model2.decoder1(z)
        generated_images2 = x2.view(nrCols, 3, SIZE, SIZE)
        generated_images2 -= 0.5

    return generated_images1, generated_images2

argv = sys.argv
argc = len(argv)

print('%s generates images using HVAE model' % argv[0])
print('[usage] python %s <trained paramL(.pth)> <trained paramsH(.pth)> [<SCALE(3)>]' % argv[0])

if argc < 3:
    quit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model1 = VAE(input_dim, hidden_dim, latent_dim)
model2 = VAE(input_dim, hidden_dim, latent_dim)

trained_filename1 = argv[1]
trained_filename2 = argv[2]

if argc > 3:
    SCALE = int(argv[3]) 

model1.load_state_dict(torch.load(trained_filename1, map_location=device))
model2.load_state_dict(torch.load(trained_filename2, map_location=device))

seed0 = torch.randn(1, latent_dim)

key = -1

print()
print('Press +/- key to up/down HPF effect')
print('Press ESC-key to terminate')
print()

running = True

while running:

    seed1 = torch.randn(1, latent_dim)
    imgs1, imgs2 = generate_images(model1, model2, seed0, seed1)

    for c in range(1, nrCols - 1):
        img1 = imgs1[c]
        #img = np.transpose(img,(1,2,0))
        img1 = img1.reshape((SIZE * SIZE * 3))
        img1 = img1.reshape((SIZE, SIZE, 3))
        screen1 = img1.detach().numpy()

        img2 = imgs2[c]
        #img = np.transpose(img,(1,2,0))
        img2 = img2.reshape((SIZE * SIZE * 3))
        img2 = img2.reshape((SIZE, SIZE, 3))
        screen2 = img2.detach().numpy()
        screen = cv2.resize(screen1 + screen2 * HPF_SCALE, (SIZE * SCALE, SIZE * SCALE))

        cv2.imshow('screen', screen)
        key = cv2.waitKey(WAIT_TIME)
        if key == 27: # ESC
            running = False
            break

        elif key == ord('+'):
            HPF_SCALE += 1
            if HPF_SCALE > HPF_SCALE_MAX:
                HPF_SCALE = HPF_SCALE_MAX
            print(HPF_SCALE)

        elif key == ord('-'):
            HPF_SCALE -= 1
            if HPF_SCALE < HPF_SCALE_MIN:
                HPF_SCALE = HPF_SCALE_MIN
            print(HPF_SCALE)

    seed0 = seed1.detach().clone()

cv2.destroyAllWindows()
