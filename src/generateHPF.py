import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import cv2, os, sys
import numpy as np
from scipy.fftpack import idct

SIZE = 128
SCALE = 32
SCALE_MIN = 0
SCALE_MAX = 128

PrevScale = -1

nrRows = 6
nrCols = 8

LEFT  = 2424832
UP    = 2490368
RIGHT = 2555904
DOWN  = 2621440

# hyperparameters
input_dim = SIZE * SIZE * 3
hidden_dim = 50
latent_dim = 10

z2 = torch.randn(nrRows * nrCols, latent_dim)

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

def generate_images(model1, model2): 

    global z2

    screenL = np.empty((SIZE * nrRows, SIZE * nrCols, 3), np.float32)
    screenH = np.empty((SIZE * nrRows, SIZE * nrCols, 3), np.float32)
    
    if not keepTopLeft:
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

        if fRandom:
            z2 = torch.randn(nrRows * nrCols, latent_dim)

        z1_hat = model1.decoder2(z2)
        z1 = reparameterize(z1_hat, torch.ones_like(z1_hat))
        x1 = model1.decoder1(z1)
        generated_images1 = x1.view(nrRows * nrCols, 3, SIZE, SIZE)

        z2_hat = model2.decoder2(z2)
        z2 = reparameterize(z2_hat, torch.ones_like(z2_hat))
        x2 = model2.decoder1(z2)
        generated_images2 = x2.view(nrRows * nrCols, 3, SIZE, SIZE)

    for R in range(nrRows):
        y = R * SIZE
        for C in range(nrCols):
            x = C * SIZE
            img1 = generated_images1[R * nrCols + C]
            #img = np.transpose(img,(0,1,2))
            img1 = img1.reshape((SIZE * SIZE * 3))
            img1 = img1.reshape((SIZE, SIZE, 3))
            img1 = img1.numpy()

            img2 = generated_images2[R * nrCols + C]
            #img = np.transpose(img,(0,1,2))
            img2 = img2.reshape((SIZE * SIZE * 3))
            img2 = img2.reshape((SIZE, SIZE, 3))
            img2 = img2.numpy()

            img2 -= 0.5

            screenL[y:y+SIZE,x:x+SIZE,:] = img1
            screenH[y:y+SIZE,x:x+SIZE,:] = img2

    return screenL, screenH

argv = sys.argv
argc = len(argv)

print('%s generates images using HVAE model' % argv[0])
print('[usage] python %s <trained param1(.pth)> <trained params2(.pth)>' % argv[0])

if argc < 3:
    quit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model1 = VAE(input_dim, hidden_dim, latent_dim)
model2 = VAE(input_dim, hidden_dim, latent_dim)

trained_filename1 = argv[1]
trained_filename2 = argv[2]

model1.load_state_dict(torch.load(trained_filename1, map_location=device))
model2.load_state_dict(torch.load(trained_filename2, map_location=device))

key = -1
no = 1
scrNo = 1

print()
print('Press ESC-key to terminate')
print('Press Arrow-keys to move cursor')
print('Press r-key to randome mode')
print('Press i-key to interpolate mode')
print('Press s-key to save the selected image')
print('Press S-key to save screen image')
print('Press +/- key to up/down HPF effect')
print('Press other key to generate images')
print()

keepTopLeft = False
fRandom = True

while key != 27:

    screenL, screenH = generate_images(model1, model2)
    cv2.imshow('screen', screenL+screenH * SCALE)

    r = 0
    c = 0

    prevR = -1
    prevC = -1

    while True:

        key = cv2.waitKeyEx(100)
        #key = cv2.waitKeyEx(0)
    
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
            selected = screenL[top:bottom, left:right] + screenH[top:bottom, left:right] * SCALE
            dst_path = '%04d.png' % no
    
            while os.path.isfile(dst_path):
                no += 1
                dst_path = '%04d.png' % no
    
            cv2.imwrite(dst_path, selected * 255)
            print('save %s' % dst_path)
  
        elif key == ord('S'):

            dst_path = 'screen_%04d.png' % scrNo

            while os.path.isfile(dst_path):
                scrNo += 1
                dst_path = 'screen_%04d.png' % scrNo

            cv2.imwrite(dst_path, screenL * 255 + screenH * SCALE * 255)
            print('save %s' % dst_path)

        elif key == ord('r') or key == ord('R'):

            keepTopLeft = False
            fRandom = True
            #screen = generate_images(model1, model2)

        elif key == ord('i') or key == ord('I'):

            keepTopLeft = False
            fRandom = False
            #screen = generate_images(model1, model2)

        elif key == ord('+'):
            SCALE += 1
            if SCALE > SCALE_MAX:
                SCALE = SCALE_MAX
            #screen = generate_images(model1, model2)
            cv2.imshow('screen', screenL + screenH * SCALE)

        elif key == ord('-'):
            SCALE -= 1
            if SCALE < SCALE_MIN:
                SCALE = SCALE_MIN
            #screen = generate_images(model1, model2)
            cv2.imshow('screen', screenL + screenH * SCALE)

        elif key != -1:
            break

        if r != prevR or c != prevC:
            left = c * SIZE
            right = left + SIZE
            top = r * SIZE
            bottom = top + SIZE
     
            dst = screenL + screenH * SCALE
            dst = cv2.rectangle(dst, (left, top), (right, bottom), (255, 0, 255), 2)
            H, W = dst.shape[:2]
            cv2.imshow('screen', dst) 

            Prev_Scale = SCALE

cv2.destroyAllWindows()
