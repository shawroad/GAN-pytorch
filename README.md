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
     
     组成部分
     1.编码器
       所谓编码器其实就是几层神经网络叠在一起，将一张(batch_size,28,28)图片变成(batch_size,10)向量
     2.解码器
       解码器就是将原来编码生成的(batch_size,10)向量，解码还原成(batch_size,28,28)图片
     3.符合正太分布(np.random.normal)+判别器D
       就是让编码器的编码规律符合正太分布
### 2-ACGAN
     具体的思路
     生成模型: 输入噪声和标签数据 用一系列卷积得到我们需要的图片大小
     判别模型: 预测当前图片的标签以及真假
     adversarial_loss() 判别真假损失(二分类)， auxiliary_loss()预测标签的损失(多分类) 
     
     生成模型的损失函数
     通过随机标签和随机噪声 生成一张图片 放入判别模型中判别得到生成模型的损失
     g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
     
     判别模型的损失函数 
     将真实图片和生成的假图片同时送入判别模型 训练判别模型
     d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
     d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2
     d_loss = (d_real_loss + d_fake_loss) / 2
### 3-BEGAN
   [原理讲解部分](https://blog.csdn.net/linmingan/article/details/79912988)
   ![BEGANLoss](https://github.com/shawroad/GAN-pytorch/blob/master/assert/BEGAN_Loss.png)
  
### 4-BGAN
     具体的思路
### 5-BycicleGAN
     具体的思路
     模型结构图
   ![BycicleGAN模型](https://github.com/shawroad/GAN-pytorch/blob/master/assert/bicyclegan_architecture.jpg)
     
