"""

@file  : 004-BGAN.py

@author: xiaolu

@time  : 2019-12-02

"""
import argparse
import os
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from keras.datasets import mnist
from keras.utils import to_categorical
from torch.utils.data.dataset import Dataset


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
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


def boundary_seeking_loss(y_pred, y_true):
    """
    Boundary seeking loss.
    Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
    """
    return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)


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
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    opt = parser.parse_args()
    print(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    discriminator_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # 我们加载keras中的内置数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 1, 28, 28)) / 255.
    x_train = x_train.astype(np.float32)
    y_train = to_categorical(y_train)
    # y_train = y_train.reshape(-1, 1)
    y_train = y_train.astype(np.int64)

    dxt = DataTxt()

    trainloader = torch.utils.data.DataLoader(
        dxt,
        batch_size=128,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(trainloader):

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(torch.FloatTensor(imgs))

            #  Train Generator
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = boundary_seeking_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            #  Train Discriminator
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = discriminator_loss(discriminator(real_imgs), valid)
            fake_loss = discriminator_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(trainloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(trainloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
