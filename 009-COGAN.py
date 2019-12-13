"""

@file  : 009-COGAN.py

@author: xiaolu

@time  : 2019-12-03

"""
import argparse
import os
import numpy as np
import math
import scipy
import itertools
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
from keras.datasets import mnist
from keras.utils import to_categorical


def weights_init_normal(m):
    '''
    权重初始化
    :param m:
    :return:
    '''
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class CoupledGenerators(nn.Module):
    '''
    生成网络
    相当于用噪声生成两张图片，两个管道　前段共享　后端各走各的路
    '''
    def __init__(self):
        super(CoupledGenerators, self).__init__()

        self.init_size = opt.img_size // 4

        self.fc = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.shared_conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
        )

        self.G1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

        self.G2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return img1, img2


class CoupledDiscriminators(nn.Module):
    '''
    判别网络
    同样是共享前半部分，最后一部分的全连接各自独有
    '''
    def __init__(self):
        super(CoupledDiscriminators, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.shared_conv = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.D1 = nn.Linear(512, 1)
        self.D2 = nn.Linear(512, 1)
        # self.D1 = nn.Linear(128 * ds_size ** 2, 1)
        # self.D2 = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img1, img2):
        # Determine validity of first image
        out = self.shared_conv(img1)
        # print(out.size())    # torch.Size([128, 128, 2, 2])
        out = out.view(out.shape[0], -1)
        # print(out.size())   # torch.Size([128, 512])
        validity1 = self.D1(out)
        # print(validity1.size())   # torch.Size([128, 1])

        # Determine validity of second image
        out = self.shared_conv(img2)
        out = out.view(out.shape[0], -1)
        validity2 = self.D2(out)

        return validity1, validity2


class DataTxt(Dataset):
    '''
    数据集的加载
    '''
    def __init__(self):
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, item):
        self.x = torch.from_numpy(self.x_data[item])
        self.y = torch.from_numpy(self.y_data[item])
        return self.x, self.y

    def __len__(self):
        return self.x_data.shape[0]


if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
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
    y_train = to_categorical(y_train)

    txt = DataTxt()
    dataloader1 = DataLoader(
        txt,
        batch_size=128,
        shuffle=True
    )
    dataloader2 = DataLoader(
        txt,
        batch_size=128,
        shuffle=True
    )

    # Loss function
    adversarial_loss = torch.nn.MSELoss()

    # Initialize models
    coupled_generators = CoupledGenerators()
    coupled_discriminators = CoupledDiscriminators()

    # Initialize weights
    coupled_generators.apply(weights_init_normal)
    coupled_discriminators.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(coupled_generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(coupled_discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Training
    for epoch in range(opt.n_epochs):
        for i, ((imgs1, _), (imgs2, _)) in enumerate(zip(dataloader1, dataloader2)):
            # print(imgs1.size())   # torch.Size([128, 1, 28, 28])
            # print(imgs2.size())   # torch.Size([128, 1, 28, 28])
            batch_size = imgs1.shape[0]

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            imgs1 = Variable(imgs1.type(torch.FloatTensor).expand(imgs1.size(0), 1, opt.img_size, opt.img_size))
            imgs2 = Variable(imgs2.type(torch.FloatTensor))

            #  Train Generators
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

            # Generate a batch of images
            gen_imgs1, gen_imgs2 = coupled_generators(z)
            # print(gen_imgs1.size())   # torch.Size([128, 3, 28, 28])
            # print(gen_imgs2.size())   # torch.Size([128, 3, 28, 28])

            # Determine validity of generated images
            validity1, validity2 = coupled_discriminators(gen_imgs1, gen_imgs2)

            # 生成模型的损失函数
            g_loss = (adversarial_loss(validity1, valid) + adversarial_loss(validity2, valid)) / 2

            g_loss.backward()
            optimizer_G.step()

            #  Train Discriminators
            optimizer_D.zero_grad()

            # Determine validity of real and generated images
            validity1_real, validity2_real = coupled_discriminators(imgs1, imgs2)

            validity1_fake, validity2_fake = coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())
            # print(validity1_fake.size())   # torch.Size([128, 1])
            # print(validity2_fake.size())   # torch.Size([128, 1])

            d_loss = (
                adversarial_loss(validity1_real, valid)
                + adversarial_loss(validity1_fake, fake)
                + adversarial_loss(validity2_real, valid)
                + adversarial_loss(validity2_fake, fake)
            ) / 4

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader1), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader1) + i
            if batches_done % opt.sample_interval == 0:
                gen_imgs = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
                save_image(gen_imgs, "images/%d.png" % batches_done, nrow=8, normalize=True)