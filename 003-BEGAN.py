"""

@file  : 003-BEGAN.py

@author: xiaolu

@time  : 2019-12-02

"""
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from keras.datasets import mnist
from keras.utils import to_categorical
from torch.utils.data.dataset import Dataset


def weights_init_normal(m):
    '''
    权重初始化
    :param m:
    :return:
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    '''
    生成模型
    '''
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        # 直接将隐层的维度通过全连接搞成 channel_size * init_size * init_size  (这里的init_size代表的是长或着宽)
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)  # [batch_size, channel_size, width, height]
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    '''
    判别模型
    '''
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

    def forward(self, img):
        # img: [128, 1, 28, 28]
        out = self.down(img)
        # print(out.size())   # torch.Size([128, 64, 14, 14])

        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        # print(out.size())  # torch.Size([128, 1, 28, 28])
        return out


class DataTxt(Dataset):
    # 写一个继承类 将数据整理成一个很好的样子  然后放进DataLoader()中去
    def __init__(self):
        self.Data = x_train
        self.Labels = y_train

    def __getitem__(self, item):
        data = torch.from_numpy(self.Data[item])
        label = torch.from_numpy(self.Labels[item])
        return data, label

    def __len__(self):
        return self.Data.shape[0]


if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
    opt = parser.parse_args()
    print(opt)

    # 我们加载keras中的内置数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 1, 28, 28)) / 255.
    x_train = x_train.astype(np.float32)
    y_train = to_categorical(y_train)
    y_train = y_train.astype(np.int64)

    dxt = DataTxt()

    trainloader = torch.utils.data.DataLoader(
        dxt,
        batch_size=128,
        shuffle=True,
    )

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    device = torch.device('cuda: 0' if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # BEGAN hyper parameters
    gamma = 0.75
    lambda_k = 0.001
    k = 0.0

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(trainloader):
            # print(imgs.shape)  # torch.Size([128, 1, 28, 28])
            # print(_.shape)   # torch.Size([128, 10])

            # Configure input
            real_imgs = Variable(torch.FloatTensor(imgs))

            #  Train Generator
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)
            # print(gen_imgs.shape)   # torch.Size([128, 1, 28, 28])

            # Loss measures generator's ability to fool the discriminator
            # 生成模型通过随机向量生成一个128x1x28x28的图片　然后通过判别模型(类似自编码器 先压缩再进行上采样)输出还是同样的规格,计算重构损失
            g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))  # 生成模型的重构损失

            g_loss.backward()
            optimizer_G.step()

            #  Train Discriminator
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            # 将真实和生成的图片送入判别模型
            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs.detach())
            # 两种重构损失  真实样本的重构损失减去虚假样本的重构损失
            d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
            d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
            d_loss = d_loss_real - k * d_loss_fake

            d_loss.backward()
            optimizer_D.step()

            # 更新虚假样本的权重
            # Update weights
            diff = torch.mean(gamma * d_loss_real - d_loss_fake)

            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

            # Update convergence metric
            M = (d_loss_real + torch.abs(diff)).item()

            # Log Progress
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                % (epoch, opt.n_epochs, i, len(trainloader), d_loss.item(), g_loss.item(), M, k)
            )

            batches_done = epoch * len(trainloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
