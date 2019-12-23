"""

@file  : 026-SoftmaxGAN.py

@author: xiaolu

@time  : 2019-12-17

"""
import argparse
import os
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data.dataset import Dataset
from keras.datasets import mnist
from keras.utils import to_categorical


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.img_size ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)

        return validity


def log(x):
    return torch.log(x + 1e-8)


class DataTxt(Dataset):
    '''
    数据加载
    '''
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __getitem__(self, item):
        self.x_ = torch.from_numpy(self.x_data[item])
        self.y_ = torch.from_numpy(self.y_data[item])
        return self.x_, self.y_

    def __len__(self):
        return self.x_data.shape[0]


if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    opt = parser.parse_args()
    print(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    devices = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 1, 28, 28)) / 255.
    x_train = x_train.astype(np.float32)
    y_train = to_categorical(y_train)
    y_train = y_train.astype(np.long)

    txt = DataTxt(x_train, y_train)
    dataloader = DataLoader(txt, batch_size=128, shuffle=True)

    # Loss function
    adversarial_loss = torch.nn.BCELoss().to(devices)

    # Initialize generator and discriminator
    generator = Generator().to(devices)
    discriminator = Discriminator().to(devices)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ==== Training ====
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            g_target = 1 / (batch_size * 2)  # 相当于生成模型训练两次 然后判别模型训练成一次
            d_target = 1 / batch_size

            # Configure input
            real_imgs = Variable(torch.FloatTensor(imgs)).to(devices)

            # Sample noise as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(devices)

            # Generate a batch of images
            gen_imgs = generator(z)

            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs)

            # Partition function
            Z = torch.sum(torch.exp(-d_real)) + torch.sum(torch.exp(-d_fake))

            # Calculate loss of discriminator and update
            d_loss = d_target * torch.sum(d_real) + log(Z)
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            # Calculate loss of generator and update
            g_loss = g_target * (torch.sum(d_real) + torch.sum(d_fake)) + log(Z)
            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)