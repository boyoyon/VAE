# https://qiita.com/hkthirano/items/7381095aaee668513487
import torch
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
from statistics import mean
import sys
import numpy as np

SIZE = 64

batch_size = 16

latent_dim = 1024

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            #nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).squeeze()

startEpoch = 1

argv = sys.argv
argc = len(argv)

print('%s trains DCGAN model' % argv[0])
print('[usage] python %s <train images(.npy)>' % argv[0])

if argc < 2:
    quit()

train_images = np.load(argv[1])
train_images = np.transpose(train_images, (0, 3, 1, 2))
nrData, H, W, C = train_images.shape
print('train_images:', train_images.shape)

train_images = train_images.astype(np.float32) / 255.0
dataset = torch.from_numpy(train_images).clone()

transform = transforms.Resize(size=(SIZE, SIZE))
dataset = transform(dataset)

data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: %s' % device)


model_G = Generator().to(device)
model_D = Discriminator().to(device)

optimizer_G = optim.Adam(model_G.parameters(),
    lr=0.0002, betas=(0.5, 0.999))

optimizer_D = optim.Adam(model_D.parameters(),
    lr=0.0002, betas=(0.5, 0.999))

# ロスを計算するときのラベル変数
ones = torch.ones(batch_size).to(device) # 正例 1
zeros = torch.zeros(batch_size).to(device) # 負例 0
loss_f = nn.BCEWithLogitsLoss()

# 途中結果の確認用の潜在特徴z
check_z = torch.randn(batch_size, latent_dim, 1, 1).to(device)


# 訓練関数
def train_dcgan(model_G, model_D, optimizer_G, optimizer_D, data_loader):
    log_loss_G = []
    log_loss_D = []
    
    for real_img in data_loader:
        batch_len = len(real_img)


        # == Generatorの訓練 ==
        # 偽画像を生成
        z = torch.randn(batch_len, latent_dim, 1, 1).to(device)
        fake_img = model_G(z)

        # 偽画像の値を一時的に保存 => 注(１)
        fake_img_tensor = fake_img.detach()

        # 偽画像を実画像（ラベル１）と騙せるようにロスを計算
        out = model_D(fake_img)
        loss_G = loss_f(out, ones[: batch_len])
        log_loss_G.append(loss_G.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()


        # == Discriminatorの訓練 ==
        # sample_dataの実画像
        real_img = real_img.to(device)

        # 実画像を実画像（ラベル１）と識別できるようにロスを計算
        real_out = model_D(real_img)
        loss_D_real = loss_f(real_out, ones[: batch_len])

        # 計算省略 => 注（１）
        fake_img = fake_img_tensor

        # 偽画像を偽画像（ラベル０）と識別できるようにロスを計算
        fake_out = model_D(fake_img_tensor)
        loss_D_fake = loss_f(fake_out, zeros[: batch_len])

        # 実画像と偽画像のロスを合計
        loss_D = loss_D_real + loss_D_fake
        log_loss_D.append(loss_D.item())

        # 微分計算・重み更新 => 注（２）
        model_D.zero_grad()
        model_G.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    return mean(log_loss_G), mean(log_loss_D)

for epoch in range(startEpoch, 10000):
    loss_G, loss_D = train_dcgan(model_G, model_D, optimizer_G, optimizer_D, data_loader)

    print('%d/%d' % (epoch+1, 10000))
    print('loss_G:', loss_G)
    print('loss_D:', loss_D)


    # 訓練途中のモデル・生成画像の保存

    if (epoch+1) % 10 == 0:
        torch.save(model_G.state_dict(), 'modelG_%04d.pth' % (epoch+1))
        torch.save(model_D.state_dict(), 'modelD_%04d.pth' % (epoch+1))
        torch.save(optimizer_G.state_dict(), 'optimizerG_%04d.pth' % (epoch+1))
        torch.save(optimizer_D.state_dict(), 'optimizerD_%04d.pth' % (epoch+1))
