# GAN-pytorch
Implementation of GAN

### 1-AAE
     具体的思路:采用的类似变分自编码器的思想，训练生成模型的损失分为为两部分: 原始变分自编码器的重构损失和隐层输出经过判别模型的输出(真假)损失. 训练判别模型的损失也分为两部分: 生成随机噪声(大小是变分自编码器的中间层大小)判断为真的损失和 真实图片经过自编码器得到隐层判断为假的损失。
     生成模型的损失函数: 
     g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(decoded_imgs, real_imgs)
     判别模型的损失函数:
     real_loss = adversarial_loss(discriminator(z), valid)
     fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
     d_loss = 0.5 * (real_loss + fake_loss)
### 2-ACGAN
     具体的思路:
