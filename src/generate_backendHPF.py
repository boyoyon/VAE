import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import cv2, os, sys
import numpy as np

SIZE = 128

ALPHA = 0.2
ALPHA_MAX = 1.0
ALPHA_MIN = 0.0

CENTER_COEFF = 25
kernel = np.array([[-2.0, -4.0, -2.0], [-4.0, CENTER_COEFF, -4.0], [-2.0, -4.0, -2.0]])

nrRows = 5
nrCols = 5

LEFT  = 2424832
UP    = 2490368
RIGHT = 2555904
DOWN  = 2621440

# hyperparameters
input_dim = SIZE * SIZE * 3
hidden_dim = 50
latent_dim = 10

z2 = torch.randn(nrRows * nrCols, latent_dim)

fRandom = True

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

    screen1 = np.empty((SIZE * nrRows, SIZE * nrCols, 3), np.float32)
    screen2 = np.empty((SIZE * nrRows, SIZE * nrCols, 3), np.float32)
    
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
            screen1[y:y+SIZE,x:x+SIZE,:] = img

    screen2 = cv2.filter2D(screen1, -1, kernel)
    
    return screen1, screen2

def main():

    global fRandom, ALPHA 

    argv = sys.argv
    argc = len(argv)
    
    print('%s generates images using HVAE model' % argv[0])
    print('[usage] python %s [<trained parameters(.pth)>]' % argv[0])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    model = VAE(input_dim, hidden_dim, latent_dim)
    
    pre_trained_filename = os.path.join(os.path.dirname(__file__), '..\\data\\vae_1120.pth')
    
    if argc > 1:
        pre_trained_filename = argv[1]
    
    if not os.path.isfile(pre_trained_filename):
        print('trained parameter is not fount at %s' % pre_trained_filename)
        quit()
    
    model.load_state_dict(torch.load(pre_trained_filename, map_location=device))
    
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
    
    key = -1
    no = 1
    scrNo = 1
    
    r = 0
    c = 0
    
    prevR = -1
    prevC = -1
    
    screen1, screen2 = generate_images(model)

    fUpdate = True

    while key != 27:
    
        key = cv2.waitKeyEx(100)
    
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
            selected = screen[top:bottom, left:right]
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

            cv2.imwrite(dst_path, screen * 255)
            print('save %s' % dst_path)

        elif key == ord('r') or key == ord('R'):

            fRandom = True
            fUpdate = True
            screen1, screen2 = generate_images(model)

        elif key == ord('i') or key == ord('I'):

            fRandom = False
            fUpdate = True
            screen1, screen2 = generate_images(model)

        elif key == ord('+'):
            ALPHA += 0.1
            if ALPHA > ALPHA_MAX:
                ALPHA = ALPHA_MAX
            print(ALPHA)
            cv2.imshow('screen', screen1 * (1.0 - ALPHA) + screen2 * ALPHA)

        elif key == ord('-'):
            ALPHA -= 0.1
            if ALPHA < ALPHA_MIN:
                ALPHA = ALPHA_MIN
            print(ALPHA)
            cv2.imshow('screen', screen1 * (1.0 - ALPHA) + screen2 * ALPHA)

        elif key != -1:
            fUpdate = True
            screen1, screen2 = generate_images(model)

        if r != prevR or c != prevC or fUpdate:
            left = c * SIZE
            right = left + SIZE
            top = r * SIZE
            bottom = top + SIZE
     
            dst = screen1 * (1.0 - ALPHA) + screen2 * ALPHA
            dst = cv2.rectangle(dst, (left, top), (right, bottom), (255, 0, 255), 2)
            H, W = dst.shape[:2]
            cv2.imshow('screen', dst) 
            
            prevR = r
            prevC = c
            fUpdate = False

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

