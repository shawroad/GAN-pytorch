"""

@file  : 007-CGAN.py

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
    '''
    生成模型
    '''
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)  # 对类别进行词嵌入(相当于一个全连接映射)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)   # 将噪声和标签的嵌入向量进行拼接

        img = self.model(gen_input)

        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    '''
    将图片维度压平和
    '''
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        # 判别模型也是将图片和标签拼接到一块进行判别
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(torch.LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


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
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # 加载keras中内置的数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 1, 28, 28)) / 255.
    x_train = x_train.astype(np.float32)

    y_train = to_categorical(y_train).astype(np.long)

    dxt = DataTxt()

    dataloader = torch.utils.data.DataLoader(
        dxt,
        batch_size=128,
        shuffle=True,
    )

    devices = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    #  Training
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            # print(imgs.size())    # torch.Size([128, 1, 28, 28])
            # print(labels.size())   # torch.Size([128, 10])

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(torch.FloatTensor(imgs))
            labels = Variable(torch.LongTensor(labels))

            #  Train Generator
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            labels = torch.topk(labels, 1)[1].squeeze()   # 把one_hot还原成普通的标签

            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(torch.LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            # print(gen_imgs.size())

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            #  Train Discriminator
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)