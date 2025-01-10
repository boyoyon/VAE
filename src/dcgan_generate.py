# https://qiita.com/hkthirano/items/7381095aaee668513487
import torch
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
import tqdm
from statistics import mean
import numpy as np
import cv2, os, sys

SIZE = 64

nrRows = 5
nrCols = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: %s' % device)

# 潜在特徴100次元ベクトルz
latent_dim = 1024

z = torch.randn(nrRows * nrCols, latent_dim, 1, 1).to(device)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

def generate_images(model):

    global z

    if fRandom:
        z = torch.randn(nrRows * nrCols, latent_dim, 1, 1).to(device)
    
    else:
        z4 = torch.randn(4, latent_dim, 1, 1).to(device)

        for r in range(nrRows):
            r2 = nrRows - 1 - r
            z[r * nrCols]              = (z4[0] * r2 + z4[1] * r) / (nrRows - 1)
            z[r * nrCols + nrCols - 1] = (z4[2] * r2 + z4[3] * r) / (nrRows - 1)

        for r in range(nrRows):
            left = z[r * nrCols]
            right = z[r * nrCols + nrCols - 1]
            for c in range(nrCols):
                c2 = nrCols - 1 - c
                z[r * nrCols + c] = (left * c2 + right * c) / (nrCols - 1)

    fake_img = model_G(z)
    
    generated_images = fake_img.cpu().detach().numpy()
    generated_images = np.transpose(generated_images, (0, 2, 3, 1)) 
    generated_images = np.clip(generated_images, 0, 1)
    generated_images *= 255
    generated_images = np.clip(generated_images, 0, 255)
    generated_images = generated_images.astype(np.uint8)

    return generated_images

argv = sys.argv
argc = len(argv)

print('%s generates image using DCGAN model' % argv[0])
print('[usage] python %s <parameter file>' % argv[0])

if argc < 2:
    quit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:%s' % device)

model_G = Generator().to(device)
model_G.load_state_dict(torch.load(argv[1], map_location=device))

key = -1
print('Hit ESC-key to terminate')

fRandom = True

no = 1

while key != 27:

    generated_images = generate_images(model_G)

    screen = np.empty((nrRows * SIZE, nrCols * SIZE, 3), np.uint8)
    
    idx = 0
    for r in range(nrRows):
        top = r * 64
        bottom = top + 64
        for c in range(nrCols):
            left = c * 64
            right = left + 64
            #screen[top:bottom, left:right, :] = cv2.cvtColor(generated_images[idx], cv2.COLOR_RGB2BGR)
            screen[top:bottom, left:right, :] = generated_images[idx]
            idx += 1
   
    screen2 = cv2.resize(screen, (nrCols * SIZE * 2, nrRows * SIZE * 2))
    cv2.imshow('screen', screen2)
    key = cv2.waitKey(0)

    if key == ord('r'):
        fRandom = not fRandom

    elif key == ord('s') or key == ord('S'):
        dst_path = 'screen_%04d.png' % no
        while os.path.isfile(dst_path):
            no += 1
            dst_path = 'screen_%04d.png' % no

        cv2.imwrite(dst_path, screen2)
        print('save %s' % dst_path)
        no += 1

cv2.destroyAllWindows()

