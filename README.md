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
        生成模型部分: 将图片随机mask掉一小部分，mask掉的地方填充为1 然后生成模型是将mask后的图片送入模型，然后得到真实的mask那
    部分。计算一个重构误差。
        判别模型部分: 将生成的图片，和真实图片送进去判别。
     损失函数:    
        生成模型损失: 两部分，第一部分是预测mask掉那部分的重构误差，第二部分是判别模型将生成的图片判断为真的损失
        判别模型损失: 将生成图片判断为假，将真实图片判断为真的损失。
### 11-CycleGAN
     具体的思路:
     生成模型的损失:
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)   # L1_loss
            loss_id_B = criterion_identity(G_AB(real_B), real_B)   # L1_loss
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)   # 通过realA生成fakeB
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)   # 通过realB生成fakeA
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)    # 接着有通过fakeB去重构A
            loss_cycle_A = criterion_cycle(recov_A, real_A)  # 算重构A和真实A的损失
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
       判别模型的损失:
            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2   # 判别模型A的损失
   ![Cycle模型结构](https://github.com/shawroad/GAN-pytorch/blob/master/assert/CycleGAN.png)

### 12-DCGAN
     具体的思路:
        不采用任何池化层（ Pooling Layer ）， 在判别器D 中，用带有步长（ Stride)的卷积来代替池化层。
        在G 、D 中均使用Batch Normalization帮助模型收敛。
        在G中，激活函数除了最后一层都使用ReLU 函数，而最后一层使用tanh函数。使用tanh函数的原因在于最后一层要输出图像，而图像的
     像素值是有一个取值范围的，如0～255 。ReLU函数的输出可能会很大，而tanh函数的输出是在-1～1之间的，只要将tanh函数的输出加1再
     乘以127.5可以得到0～255 的像素值。
        在D 中，激活函数都使用Leaky ReLU作为激活函数。
### 13-DiscoGAN
     具体的思路:


### 14-DraGAN
     具体的思路:


### 15-DualGAN
     具体的思路:

### 16-EbGAN
     具体的思路:

### 17-EsrGAN
     具体的思路:

### 18-GAN
     具体的思路:
     此思路略去。。。。。

### 19-InfoGAN
     具体的思路:
         生成模型部分: 总共有三个输入(标签用one_hot表示作为输入, 随机向量latent_dim, 额外信息code 两维的数据)　将输入进行拼接　然后
     输入到模型中，生成我们需要的大小图片。　　
         判别模型部分: 将真，假图片输入，得到以上三个输出。
     
     生成模型的损失: 让判别模型把假判别为真　来训练生成模型
     判别模型的损失: 将假图判断为假，　将真图判断为真，　两种损失平均  训练判别模型
     联合损失: 两种损失（标签预测损失，　额外信息预测损失）　同时训练判别模型和生成模型。

### 20-LsGAN
     具体的思路
     　　和原始GAN的结构相同。　唯一不同的是，损失函数进行了改变。　原始GAN的损失用的是交叉损失BCELoss  
     　LsGAN的损失用的是均方误差MSELoss

### 21-munit
     具体的思路
   　　  

### 22-Pix2Pix
     具体的思路
     　　首先有一一对应的图片，也就是没有上色和上色的 　
     　　生成模型部分: 用realA去生成fakeB　
     　　判别模型部分: realA和realB进行拼接　然后送入判别模型，最后的输出是一个小区域。　如: 5x5的一个小框
     　　生成模型损失: 两部分损失，一: 生成的fakeB和realB计算一个重构误差损失.　二: 将fakeB和realA进行拼接　
       放入判别模型，判别为真(这里的损失要注意是什么东西)
     　　判别模型损失: 两部分损失，一: 将fakeB和realA进行拼接放入判别模型　判别为假的损失　将realB和realA放
       进判别模型，判别为真的损失。
     　　生成模型损失: 两部分损失，一: 生成的fakeB和realB计算一个重构误差损失.　二: 将fakeB和realA进行拼接　
### 23-pixlda
     具体的思路

### 24-RelativisticGAN
     具体的思路

### 25-SGAN
     具体的思路
### 26-SoftmaxGAN
     具体的思路
     
