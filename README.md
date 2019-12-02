# GAN-pytorch
Implementation of GAN

### 1-AAE
     具体的思路
     采用的类似变分自编码器的思想，训练生成模型的损失分为为两部分: 原始变分自编码器的重构损失和隐层输出经过判别模型的输出(真假)损失. 训练判别模型的损失也分为两部分: 生成随机噪声(大小是变分自编码器的中间层大小)判断为真的损失和 真实图片经过自编码器得到隐层判断为假的损失。
     生成模型的损失函数: 
     g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(decoded_imgs, real_imgs)
     判别模型的损失函数:
     real_loss = adversarial_loss(discriminator(z), valid)
     fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
     d_loss = 0.5 * (real_loss + fake_loss)
     组成部分:
     1.编码器
       所谓编码器其实就是几层神经网络叠在一起，将一张(batch_size,28,28)图片变成(batch_size,10)向量
     2.解码器
       解码器就是将原来编码生成的(batch_size,10)向量，解码还原成(batch_size,28,28)图片
     3.符合正太分布(np.random.normal)+判别器D
       就是让编码器的编码规律符合正太分布
### 2-ACGAN
     具体的思路:
### 3-BEGAN
     具体的思路:
### 4-BGAN
     具体的思路:
### 5-BycicleGAN
     具体的思路:
     
