"""

@file  : 010-Context_Encoder.py

@author: xiaolu

@time  : 2019-12-04

"""
import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


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
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            '''
            下采样
            :param in_feat: 输入的通道
            :param out_feat: 输出的通道
            :param normalize: 是否进行归一化
            :return:
            '''
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            '''
            上采样
            :param in_feat: 输入的通道
            :param out_feat: 输出的通道
            :param normalize: 是否进行归一化
            :return:
            '''
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            nn.Conv2d(512, 4000, 1),
            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    '''
    判别模型
    '''
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob('%s/*.jpg' % root))
        self.files = self.files[:-40] if mode == 'train' else self.files[-40:]

    def apply_random_mask(self, img):
        '''
        对图片进行随机mask
        :param img:
        :return: masked_img: mask后的图片  masked_part: 被mask掉的那部分 也就是原始图像中那部分
        '''
        # 选一个随机的点 然后构造一个正方形 然后将其mask掉
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1: y2, x1: x2]
        masked_img = img.clone()
        masked_img[:, y1: y2, x1: x2] = 1  # mask掉的地方用1进行填充
        return masked_img, masked_part

    def apply_center_mask(self, img):
        '''
        mask掉图片的中间部分
        :param img:
        :return:
        '''
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i: i + self.mask_size, i: i + self.mask_size] = 1
        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            # 对于训练图片 我们进行随机mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # 对于测试图片 我们进行中间mask
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)


def save_sample(batchs_done):
    samples, masked_samples, i = next(iter(test_dataloader))
    samples = Variable(torch.FloatTensor(samples))
    masked_samples = Variable(torch.FloatTensor(masked_samples))
    i = i[0].item()
    gen_mask = generator(masked_samples)

    filled_samples = masked_samples.clone()
    filled_samples[:, :, i: i + opt.mask_size, i: i + opt.mask_size] = gen_mask

    # 保存
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, 'images/%d.png' % batches_done, nrow=6, normalize=True)


if __name__ == '__main__':

    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    # Calculate output of image discriminator (PatchGAN)
    patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
    patch = (1, patch_h, patch_w)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    pixelwise_loss = torch.nn.L1Loss()

    # Initialize generator and discriminator
    generator = Generator(channels=opt.channels)
    discriminator = Discriminator(channels=opt.channels)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Dataset loader
    transforms_ = [
        transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        ImageDataset("./data/train/", transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=1
    )
    test_dataloader = DataLoader(
        ImageDataset("./data/test/", transforms_=transforms_, mode="val"),
        batch_size=12,
        shuffle=True,
        num_workers=1
    )
    # for a, b, c in dataloader:
    #     print(a.size())   # torch.Size([8, 3, 128, 128])
    #     print(b.size())   # torch.Size([8, 3, 128, 128])
    #     print(c.size())   # torch.Size([8, 3, 64, 64])
    #     exit()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    #  Training
    for epoch in range(opt.n_epochs):
        for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = Variable(torch.FloatTensor(imgs))
            masked_imgs = Variable(torch.FloatTensor(masked_imgs))
            masked_parts = Variable(torch.FloatTensor(masked_parts))

            #  Train Generator
            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_parts = generator(masked_imgs)   # 通过mask后填充的图片去预测mask掉的部分
            # print(gen_parts.size())   # torch.Size([8, 3, 64, 64])

            # Adversarial and pixelwise loss
            g_adv = adversarial_loss(discriminator(gen_parts), valid)  # 对生成模型的训练　认为生成的是有效的
            g_pixel = pixelwise_loss(gen_parts, masked_parts)   # 生成的和真实的计算误差
            # Total loss
            g_loss = 0.001 * g_adv + 0.999 * g_pixel

            g_loss.backward()
            optimizer_G.step()

            #  Train Discriminator
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
            )

            # Generate sample at sample interval
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_sample(batches_done)