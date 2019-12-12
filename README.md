# GAN-pytorch
Implementation of GAN

### 1-AAE
     具体的思路
       采用的类似变分自编码器的思想，训练生成模型的损失分为为两部分: 原始变分自编码器的重构损失和隐层输出经过判别模型的输出(真假)损失. 
     训练判别模型的损失也分为两部分: 生成随机噪声(大小是变分自编码器的中间层大小)判断为真的损失和 真实图片经过自编码器得到隐层判断为
     假的损失。
     
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
        BGAN的全称是Boundary Seeking GAN，它的中文翻译是：基于边界寻找的gan，那么这个边界指的是谁呢？一般而言，判别器的loss稳定
     在0.5的时候，生成图片的效果是最好的，而这个边界指代的就是0.5。　
### 5-BycicleGAN
     具体的思路:
        有两组图片，记做realA和realB, 首先，我们将realB送入encoder中,得到方差的对数和均值, 然后通过重参数的技巧得到一个向量，将这
     个向量和图片realA合并，送入生成模型(U-net)中生成图片，　接着将生成图片送入判别模型中。　另外，还自动生成一个隐藏向量，然后和
     realA组合送入生成模型，然后将生成的图片送入到判别模型中。　encoder模型的损失包括: realB和fakeB的损失， KL损失，　判别模型
     将fakeB判别为真的损失，　判别模型将随机向量生成的图片判别为真的损。　generation模型的损失包括: 随机向量送入encoder中得到的
     均值向量和随机向量的mae_loss。　decision模型的损失包括: 将realB判别为真的损失, 将fakeB判别为假的损失，
     
     模型结构图
   ![BycicleGAN模型](https://github.com/shawroad/GAN-pytorch/blob/master/assert/bicyclegan_architecture.jpg)
   
### 6-CCGAN
     具体的思路:
        CCGAN的全称是“Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks”，中文翻译是
     “基于上下文条件以及半监督学习的生成对抗网”。这个网络损失函数遵循与基本的GAN，唯一有点区别的是:在这个网络中，输入的图片有两部分，
     第一部分为被mask处理过的图片（比如一个区域设置为0）,第二部分为低分辨图片（就是将图片直接resize为一个小尺度的图片，第二部可选择
     不要，当然生成图片的效果就会变差）
### 7-CGAN
     具体的思路:
        生成模型部分: 是将标签进行嵌入 然后和随机向量进行拼接，然后输入模型 生成我们的图片。
        判别模型部分: 将我们的图片打平，然后和标签的嵌入向量进行拼接，通过一系列全连接 得到最终的真或这假
     损失函数:
        生成模型损失: 生成的图片判断为真的交叉损失
        判别模型损失: 将真实图片和我们的生成图片同时扔进模型，然后判断真假 得出交叉损失。。

### 8-ClusterGAN
     具体的思路:
     

### 9-COGAN
     具体的思路:
        生成模型部分: 将随机向量直接扔进模型　去生成两张图片，前面的卷积共享，后半段不共享，然后生成两张图片
        判别模型部分: 将生成的两张图片扔进判别模型，然后去判断真假，　也是前半部分共享，后半部分独有。
     损失函数:
        生成模型损失: 将生成的两张图片扔进判别模型，分别判断为真，得到损失　去优化生成模型
        判别模型损失: 将真实数据加载两次　两真，两假　扔进模型，真的判断为真，假的判断为假的损失　然后优化判别模型。

### 10-Context_Encoder
     具体的思路:


### 11-CycleGAN
     具体的思路:


### 12-DCGAN
     具体的思路:


### 13-DiscoGAN
     具体的思路:


### 14-DraGAN
     具体的思路:


### 15-DualGAN
     具体的思路:
     
