import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os, re, subprocess, sys

# hyperparameters
input_dim = 128 * 128 * 3
hidden_dim = 50
latent_dim = 10

epochs = 100000

#learning_rate = 1e-3
#learning_rate = 1e-4 * 0.5
learning_rate = 1e-4

#batch_size = 32
batch_size = 1024

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

nrHistory = 10

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device='cpu'):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(input_dim, hidden_dim, device = device)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim, device = device)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim, device = device)

    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, use_sigmoid=False, device='cpu'):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(latent_dim, hidden_dim, device = device)
        self.linear2 = nn.Linear(hidden_dim, output_dim, device = device)
        self.use_sigmoid = use_sigmoid

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        if self.use_sigmoid:
            h = torch.sigmoid(h)
        return h

def reparameterize(mu, sigma, device = 'cpu'):
    eps = torch.randn_like(sigma, device = device)
    z = mu + eps * sigma
    return z

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device = 'cpu'):
        super().__init__()
        self.device = device
        self.encoder1 = Encoder(input_dim, hidden_dim, latent_dim, device = device)
        self.encoder2 = Encoder(latent_dim, hidden_dim, latent_dim, device = device)
        self.decoder1 = Decoder(latent_dim, hidden_dim, input_dim, use_sigmoid=True, device = device)
        self.decoder2 = Decoder(latent_dim, hidden_dim, latent_dim, device = device)

    def get_loss(self, x):
        mu1, sigma1 = self.encoder1(x)
        z1 = reparameterize(mu1, sigma1, device = self.device)
        mu2, sigma2 = self.encoder2(z1)
        z2 = reparameterize(mu2, sigma2, device = self.device)

        z_hat = self.decoder2(z2)
        x_hat = self.decoder1(z1)

        # loss
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction='sum')
        L2 = - torch.sum(1 + torch.log(sigma2 ** 2) - mu2 ** 2 - sigma2 ** 2)
        L3 = - torch.sum(1 + torch.log(sigma1 ** 2) - (mu1 - z_hat) ** 2 - sigma1 ** 2)
        return (L1 + L2 + L3) / batch_size

def main():

    argv = sys.argv
    argc = len(argv)
    
    print('%s trains VAE model' % argv[0])
    print('[usage] python %s <training data(.npy)> [<model for restart> <optimizer for restart>]' % argv[0])
    
    if argc < 2:
        quit()
    
    # dataset
    image_train = np.load(argv[1])
    nrData, H, W, C = image_train.shape
    print(image_train.shape)
    
    image_train = image_train.astype(np.float32) / 255.0
    image_train = image_train.reshape((nrData, H * W * C)) 
    print(image_train.shape)
    
    images = torch.from_numpy(image_train).clone()
    images_gpu = images.to(device)
    
    #dataloader = torch.utils.data.DataLoader(images_gpu, batch_size=batch_size, shuffle=True)
    
    print(batch_size, learning_rate)
    
    model = VAE(input_dim, hidden_dim, latent_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    lossIdxMin = 3000

    if argc > 2:
        model.load_state_dict(torch.load(argv[2]))
        
        base = os.path.basename(argv[2])
        filename = os.path.splitext(base)[0]
        lossIdxMin = int(re.sub(r'.*_', '', filename))
    
    print('lossIdxMin:', lossIdxMin)
    
    if argc > 3:
        optimizer.load_state_dict(torch.load(argv[3]))
    
    model.to(device)
   
    models = []
    optimizers = []

    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
    
        dataloader = torch.utils.data.DataLoader(images_gpu, batch_size=batch_size, shuffle=True)
    
        for x in dataloader:
            optimizer.zero_grad()
            loss = model.get_loss(x)
            loss.backward()
            optimizer.step()
    
            loss_sum += loss.item()
            cnt += 1
    
        loss_avg = loss_sum / cnt
        print('%d/%d: %f' % (epoch+1, epochs, loss_avg))
    
        # save model(parameters)
        lossIdx = int(loss_avg)
    
        if lossIdx < lossIdxMin:
    
            dst_path = 'model_%d.pth' % lossIdx
            torch.save(model.state_dict(), dst_path)
            print('save %s' % dst_path)
            models.append(dst_path)

            dst_path = 'optimizer_%d.pth' % lossIdx
            torch.save(optimizer.state_dict(), dst_path)
            print('save %s' % dst_path)
            optimizers.append(dst_path)

            while len(models) >= nrHistory:
                cmd = 'del %s' % models[0]
                subprocess.run(cmd, shell=True)
                del(models[0])

                cmd = 'del %s' % optimizers[0]
                subprocess.run(cmd, shell=True)
                del(optimizers[0])

            lossIdxMin = lossIdx

if __name__ == '__main__':
    main()



