"""

@file  : 015-DualGAN.py

@author: xiaolu

@time  : 2019-12-06

"""
import glob
from torch.utils.data import Dataset
from PIL import Image
import argparse
import os
import numpy as np
import itertools
import torch.nn as nn
import torch
import time
import datetime
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# U-NET
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(nn.ConvTranspose2d(128, channels, 4, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        # Propogate noise through fc layer and reshape to img shape
        d1 = self.down1(x)
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


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discrimintor_block(in_features, out_features, normalize=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discrimintor_block(in_channels, 64, normalize=False),
            *discrimintor_block(64, 128),
            *discrimintor_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, kernel_size=4)
        )

    def forward(self, img):
        return self.model(img)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    validity = D(interpolates)
    fake = Variable(torch.FloatTensor(np.ones(validity.shape)), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=validity,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(torch.FloatTensor(imgs["A"]))
    fake_B = G_AB(real_A)
    AB = torch.cat((real_A.data, fake_B.data), -2)
    real_B = Variable(torch.FloatTensor(imgs["B"]))
    fake_A = G_BA(real_B)
    BA = torch.cat((real_B.data, fake_A.data), -2)
    img_sample = torch.cat((AB, BA), 0)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=8, normalize=True)


if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    # Loss function
    cycle_loss = torch.nn.L1Loss().to(device)

    # Loss weights
    lambda_adv = 1
    lambda_cycle = 10
    lambda_gp = 10

    # Initialize generator and discriminator
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    if opt.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
        G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Configure data loader
    transforms_ = [
        transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        ImageDataset("./data/%s" % opt.dataset_name, transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    val_dataloader = DataLoader(
        ImageDataset("./data/%s" % opt.dataset_name, mode="val", transforms_=transforms_),
        batch_size=16,
        shuffle=True,
        num_workers=1,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    #  ----Training----
    batches_done = 0
    prev_time = time.time()
    for epoch in range(opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Configure input
            imgs_A = Variable(torch.FloatTensor(batch["A"])).to(device)
            imgs_B = Variable(torch.FloatTensor(batch["B"])).to(device)

            #  Train Discriminators
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            # Generate a batch of images
            fake_A = G_BA(imgs_B).detach()  # 用realB去生成fakeA
            fake_B = G_AB(imgs_A).detach()  # 用realA去生成fakeB

            # Domain A
            # Compute gradient penalty for improved wasserstein training
            gp_A = compute_gradient_penalty(D_A, imgs_A.data, fake_A.data)
            # Adversarial loss
            D_A_loss = - torch.mean(D_A(imgs_A)) + torch.mean(D_A(fake_A)) + lambda_gp * gp_A

            # Domain B
            # Compute gradient penalty for improved wasserstein training
            gp_B = compute_gradient_penalty(D_B, imgs_B.data, fake_B.data)
            # Adversarial loss
            D_B_loss = -torch.mean(D_B(imgs_B)) + torch.mean(D_B(fake_B)) + lambda_gp * gp_B

            # Total loss
            D_loss = D_A_loss + D_B_loss

            D_loss.backward()
            optimizer_D_A.step()
            optimizer_D_B.step()

            if i % opt.n_critic == 0:
                # 这里规定了每个几步去训练一次判别模型
                #  Train Generators
                optimizer_G.zero_grad()

                # Translate images to opposite domain
                fake_A = G_BA(imgs_B)
                fake_B = G_AB(imgs_A)

                # Reconstruct images
                recov_A = G_BA(fake_B)
                recov_B = G_AB(fake_A)

                # Adversarial loss
                G_adv = -torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
                # Cycle loss
                G_cycle = cycle_loss(recov_A, imgs_A) + cycle_loss(recov_B, imgs_B)
                # Total loss
                G_loss = lambda_adv * G_adv + lambda_cycle * G_cycle

                G_loss.backward()
                optimizer_G.step()

                # Log Progress
                # Determine approximate time left
                batches_left = opt.n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / opt.n_critic)
                prev_time = time.time()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, cycle: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        D_loss.item(),
                        G_adv.data.item(),
                        G_cycle.item(),
                        time_left,
                    )
                )

            # Check sample interval => save sample if there
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

            batches_done += 1

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))

