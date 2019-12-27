"""

@file  : 023-pixlda.py

@author: xiaolu

@time  : 2019-12-17

"""
import argparse
import os
import numpy as np
import math
import itertools
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch
from keras.datasets import mnist
from keras.utils import to_categorical


def weights_init_normal(m):
    '''
    权重初始化
    :param m:
    :return:
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Linear(opt.latent_dim, opt.channels * opt.img_size ** 2)

        self.l1 = nn.Sequential(nn.Conv2d(opt.channels * 2, 64, 3, 1, 1), nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(opt.n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)

        self.l2 = nn.Sequential(nn.Conv2d(64, opt.channels, 3, 1, 1), nn.Tanh())

    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)

        return validity


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False), *block(64, 128), *block(128, 256), *block(256, 512)
        )

        # input_size = opt.img_size // 2 ** 4
        self.output_layer = nn.Sequential(nn.Linear(2048, opt.n_classes), nn.Softmax())

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label


class DataTxt(Dataset):
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
    parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
    parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the noise input")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes in the dataset")
    parser.add_argument("--sample_interval", type=int, default=300, help="interval betwen image samples")
    opt = parser.parse_args()
    print(opt)

    # Calculate output of image discriminator (PatchGAN)
    patch = int(opt.img_size / 2 ** 4)
    patch = (1, patch, patch)

    devices = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 1, 28, 28)) / 255.
    x_train = x_train.astype(np.float32)
    y_train = to_categorical(y_train)
    y_train = y_train.astype(np.long)

    txt = DataTxt(x_train, y_train)
    dataloader_A = DataLoader(txt, shuffle=True, batch_size=128)
    dataloader_B = DataLoader(txt, shuffle=True, batch_size=128)

    # for x, y in dataloader:
    #     print(x.size())  # torch.Size([128, 1, 28, 28])
    #     print(y.size())  # torch.Size([128, 10])
    #     exit()

    # Loss function
    adversarial_loss = torch.nn.MSELoss().to(devices)
    task_loss = torch.nn.CrossEntropyLoss().to(devices)

    # Loss weights
    lambda_adv = 1
    lambda_task = 0.1

    # Initialize generator and discriminator
    generator = Generator().to(devices)
    discriminator = Discriminator().to(devices)
    classifier = Classifier().to(devices)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    classifier.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(generator.parameters(), classifier.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    #  Training
    # Keeps 100 accuracy measurements
    task_performance = []
    target_performance = []

    for epoch in range(opt.n_epochs):
        for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):
            # print(imgs_B.size())
            # print(labels_B.size())
            batch_size = imgs_A.size(0)

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False).to(devices)
            fake = Variable(torch.FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False).to(devices)

            # Configure input
            imgs_A = Variable(torch.FloatTensor(imgs_A).expand(batch_size, 3, opt.img_size, opt.img_size)).to(devices)
            labels_A = Variable(torch.LongTensor(labels_A)).to(devices)
            imgs_B = Variable(torch.FloatTensor(imgs_B).expand(batch_size, 3, opt.img_size, opt.img_size)).to(devices)

            # ==== Train Generator ====
            optimizer_G.zero_grad()

            # Sample noise
            z = Variable(torch.FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.latent_dim)))).to(devices)

            # Generate a batch of images
            fake_B = generator(imgs_A, z)  # 通过真实图片A　和噪声的混合　生成假的B
            # print(fake_B.size())  # torch.Size([128, 3, 28, 28])

            # Perform task on translated source image
            label_pred = classifier(fake_B)
            # print(label_pred.size())  # torch.Size([128, 10])

            # Calculate the task loss
            # 注意 这里需要将原始标签从one_hot 转回来。　
            labels_A = torch.topk(labels_A, 1)[1].squeeze()
            task_loss_ = (task_loss(label_pred, labels_A) + task_loss(classifier(imgs_A), labels_A)) / 2

            # Loss measures generator's ability to fool the discriminator
            g_loss = lambda_adv * adversarial_loss(discriminator(fake_B), valid) + lambda_task * task_loss_

            g_loss.backward()
            optimizer_G.step()

            # ==== Train Discriminator ====
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(imgs_B), valid)
            fake_loss = adversarial_loss(discriminator(fake_B.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader_A),
                    d_loss.item(),
                    g_loss.item(),
                )
            )

            batches_done = len(dataloader_A) * epoch + i
            if batches_done % opt.sample_interval == 0:
                sample = torch.cat((imgs_A.data[:5], fake_B.data[:5], imgs_B.data[:5]), -2)
                save_image(sample, "images/%d.png" % batches_done, nrow=int(math.sqrt(batch_size)), normalize=True)