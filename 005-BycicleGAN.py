"""

@file  : 005-BycicleGAN.py

@author: xiaolu

@time  : 2019-12-02

"""
import torch.nn as nn
from torchvision.models import resnet18
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
import datetime
import time
import sys
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch


# 数据加载部分
class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train"):
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)


# 模型权重初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# U-NET
class UNetDown(nn.Module):
    '''
    UNet网络的下采样模块
    '''
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    '''
    UNet网络的上采样模块
    '''
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    '''
    使用Unet网络作为生成模型　造假
    '''
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape
        # 把噪声数据通过全连接整成图片的宽和高
        self.fc = nn.Linear(latent_dim, self.h * self.w)

        # 进行下采样然后进行上采样
        self.down1 = UNetDown(channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, x, z):
        # Propogate noise through fc layer and reshape to img shape
        z = self.fc(z).view(z.size(0), 1, self.h, self.w)
        d1 = self.down1(torch.cat((x, z), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)


# Encoder
class Encoder(nn.Module):
    '''
    使用resnet18对图片进行特征提取, 然后通过全连接得到均值和方差对数(类似变分自编码器)
    '''
    def __init__(self, latent_dim, input_shape):
        super(Encoder, self).__init__()
        resnet18_model = resnet18(pretrained=False)  # 解码部分我们使用resnet18网络
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])  # 不要其最后分类的三层

        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


# Discriminator
class MultiDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        channels, _, _ = input_shape

        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
                )

        self.downsample = nn.AvgPool2d(1, stride=2, padding=[1, 1], count_include_pad=False)
        #

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        # print(x.size())   # torch.Size([8, 3, 128, 128])
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            # print(m(x).size())   # torch.Size([8, 1, 8, 8])
            x = self.downsample(x)
            print(x.size())    # torch.Size([8, 3, 64, 64])
        return outputs


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    generator.eval()
    imgs = next(iter(val_dataloader))
    img_samples = None
    for img_A, img_B in zip(imgs["A"], imgs["B"]):
        # Repeat input image by number of desired columns
        real_A = img_A.view(1, *img_A.shape).repeat(opt.latent_dim, 1, 1, 1)
        real_A = Variable(torch.FloatTensor(real_A)).to(devices)
        # Sample latent representations
        sampled_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (opt.latent_dim, opt.latent_dim)))).to(devices)
        # Generate samples
        fake_B = generator(real_A, sampled_z)
        # Concatenate samples horisontally
        fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        img_sample = torch.cat((img_A, fake_B), -1)
        img_sample = img_sample.view(1, *img_sample.shape)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=8, normalize=True)
    generator.train()


def reparameterization(mu, logvar):
    '''
    重参数的技巧
    :param mu: 均值
    :param logvar: 方差对数
    :return:
    '''
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim)))).to(devices)
    z = sampled_z * std + mu
    return z


if __name__ == '__main__':
    # 超参数的定义
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--img_width", type=int, default=128, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
    parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
    parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    devices = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Loss functions
    mae_loss = torch.nn.L1Loss()

    # 把所有模型指定到对应的设备上进行训练
    # Initialize generator, encoder and discriminators
    generator = Generator(opt.latent_dim, input_shape).to(devices)
    encoder = Encoder(opt.latent_dim, input_shape).to(devices)
    D_VAE = MultiDiscriminator(input_shape).to(devices)
    D_LR = MultiDiscriminator(input_shape).to(devices)

    # 如果有与训练的模型 我们就先加载预训练好的 然后在去训练
    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        encoder.load_state_dict(torch.load("saved_models/%s/encoder_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_VAE.load_state_dict(torch.load("saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_LR.load_state_dict(torch.load("saved_models/%s/D_LR_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        D_VAE.apply(weights_init_normal)
        D_LR.apply(weights_init_normal)

    # Optimizers
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    dataloader = DataLoader(
        ImageDataset("./data/", input_shape, mode="train"),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    val_dataloader = DataLoader(
        ImageDataset("./data/", input_shape, mode="test"),
        batch_size=8,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # Adversarial loss
    valid = 1
    fake = 0
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input   一对图片
            real_A = Variable(torch.FloatTensor(batch["A"])).to(devices)
            real_B = Variable(torch.FloatTensor(batch["B"])).to(devices)

            #  Train Generator and Encoder
            optimizer_E.zero_grad()
            optimizer_G.zero_grad()

            # cVAE-GAN
            # Produce output using encoding of B (cVAE-GAN)
            mu, logvar = encoder(real_B)   # 通过编码网络得到均值和方差的对数
            encoded_z = reparameterization(mu, logvar)  # 采用重参数的技巧进行采样
            # resnet编码 最后生成均值和方差的对数,通过重参数的技巧得到随机向量, 然后送入生成模型,生成我们想要的图像
            fake_B = generator(real_A, encoded_z)   # 将采样的结果和正式的图片　生成图片
            # print(real_B.size())   # torch.Size([8, 3, 128, 128])
            # print(fake_B.size())   # torch.Size([8, 3, 128, 128])

            # Pixelwise loss of translated image by VAE
            loss_pixel = mae_loss(fake_B, real_B)    # 重构误差(生成模型的输入和输出计算出的损失)

            # Kullback-Leibler divergence of encoded B  # KL损失
            loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)

            # Adversarial loss 判别真假损失(生成的图片　强行判别为真)
            loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

            # cLR-GAN
            # Produce output using sampled z (cLR-GAN) 直接生成隐层维度(AE中间的维度)
            sampled_z = Variable(torch.FloatTensor(np.random.normal(0, 1, (real_A.size(0), opt.latent_dim)))).to(devices)
            _fake_B = generator(real_A, sampled_z)   # a图片加入噪声生成虚假的b
            # cLR Loss: Adversarial loss
            loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

            # Total Loss (Generator + Encoder)　
            loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel * loss_pixel + opt.lambda_kl * loss_kl

            loss_GE.backward(retain_graph=True)
            optimizer_E.step()

            # Generator Only Loss
            # Latent L1 loss
            _mu, _ = encoder(_fake_B)
            loss_latent = opt.lambda_latent * mae_loss(_mu, sampled_z)

            loss_latent.backward()
            optimizer_G.step()

            #  Train Discriminator (cVAE-GAN)
            optimizer_D_VAE.zero_grad()
            loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.detach(), fake)

            loss_D_VAE.backward()
            optimizer_D_VAE.step()

            #  Train Discriminator (cLR-GAN)
            optimizer_D_LR.zero_grad()

            loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(_fake_B.detach(), fake)

            loss_D_LR.backward()
            optimizer_D_LR.step()

            #  Log Progress
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D_VAE.item(),
                    loss_D_LR.item(),
                    loss_GE.item(),
                    loss_pixel.item(),
                    loss_kl.item(),
                    loss_latent.item(),
                    time_left,
                )
            )

            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(encoder.state_dict(), "saved_models/%s/encoder_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_VAE.state_dict(), "saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_LR.state_dict(), "saved_models/%s/D_LR_%d.pth" % (opt.dataset_name, epoch))
