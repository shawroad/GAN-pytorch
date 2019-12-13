"""

@file  : 019-InfoGAN.py

@author: xiaolu

@time  : 2019-12-13

"""
import argparse
import os
import numpy as np
import itertools
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from keras.datasets import mnist
import torch


def weights_init_normal(m):
    '''
    权重归一化
    :param m:
    :return:
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    '''
    生成模型
    '''
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim  # 三种数据进行相加

        self.init_size = opt.img_size // 4  # Initial size before upsampling　为了上采样两次得到标准规格图片大小的图片
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))  # 128为　初始化的通道数

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

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        # print("生成图片的规格:", img.size())
        return img


class Discriminator(nn.Module):
    '''
    判别模型
    '''
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = 2
        # Output layers　三种的输出
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)
        return validity, label, latent_code


def to_categorical(y, num_columns):
    '''
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1,
       2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
       4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5,
       6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7,
       8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    :param y:
    :param num_columns:
    :return:
    '''
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))  # y.shape[0]: 10
    y_cat[range(y.shape[0]), y] = 1.0
    return Variable(torch.FloatTensor(y_cat))


class DataTxt(Dataset):
    '''
    数据加载器
    '''
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __getitem__(self, item):
        self.x = torch.from_numpy(self.x_data[item])
        self.y = torch.from_numpy(self.y_data[item])
        return self.x, self.y

    def __len__(self):
        return self.x_data.shape[0]


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    static_sample = generator(z, static_label, static_code)
    save_image(static_sample.data, "images/static/%d.png" % batches_done, nrow=n_row, normalize=True)

    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(torch.FloatTensor(np.concatenate((c_varied, zeros), -1)))
    c2 = Variable(torch.FloatTensor(np.concatenate((zeros, c_varied), -1)))
    sample1 = generator(static_z, static_label, c1)
    sample2 = generator(static_z, static_label, c2)
    save_image(sample1.data, "images/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, "images/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)


if __name__ == '__main__':
    os.makedirs("images/static/", exist_ok=True)
    os.makedirs("images/varying_c1/", exist_ok=True)
    os.makedirs("images/varying_c2/", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
    parser.add_argument("--code_dim", type=int, default=2, help="latent code")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

    devices = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    # Loss functions
    adversarial_loss = torch.nn.MSELoss().to(devices)
    categorical_loss = torch.nn.CrossEntropyLoss().to(devices)
    continuous_loss = torch.nn.MSELoss().to(devices)

    # Loss weights
    lambda_cat = 1
    lambda_con = 0.1

    # Initialize generator and discriminator
    generator = Generator().to(devices)
    discriminator = Discriminator().to(devices)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 1, 28, 28)) / 255.
    x_train = x_train.astype(np.float32)
    y_train = y_train.reshape((-1, 1))

    txt = DataTxt(x_train, y_train)
    dataloader = DataLoader(txt, shuffle=True, batch_size=128)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    # 生成随机输入样本 100个
    static_z = Variable(torch.FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
    # 生成标签 使用one_hot编码
    static_label = to_categorical(
        np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
    )
    # 生成额外信息两维度
    static_code = Variable(torch.FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))

    # ====Training====
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            temp = labels.numpy()
            labels = torch.LongTensor(temp.reshape(-1,))

            # Configure input
            real_imgs = Variable(torch.FloatTensor(imgs))    # 真实图片
            labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)   # 将标签转为one_hot

            #  ====Train Generator====
            optimizer_G.zero_grad()
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
            code_input = Variable(torch.FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            # Generate a batch of images
            gen_imgs = generator(z, label_input, code_input)

            # Loss measures generator's ability to fool the discriminator
            validity, _, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)  # 生成模型的损失只有一种就是将fake_image 预测成　true_image
            g_loss.backward()
            optimizer_G.step()

            # ==== Train Discriminator ====
            optimizer_D.zero_grad()
            # Loss for real images
            real_pred, _, _ = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _, _ = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ==== G 和 D 同时训练
            # 重点 Information Loss
            optimizer_info.zero_grad()
            # Sample labels
            sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

            # Ground truth labels
            gt_labels = Variable(torch.LongTensor(sampled_labels), requires_grad=False)

            # Sample noise, labels and code as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
            code_input = Variable(torch.FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

            gen_imgs = generator(z, label_input, code_input)
            _, pred_label, pred_code = discriminator(gen_imgs)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
                pred_code, code_input
            )

            info_loss.backward()
            optimizer_info.step()

            # Log Progress
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)