# tensorflow-generative-model-collections
Tensorflow implementation of various GANs and VAEs.

## Generative Adversarial Networks (GANs)
### Lists  

*Name* | *Paer Link* | *Loss Function*
:---: | :---: | :--- |
**GAN** | [Arxiv](https://arxiv.org/abs/1406.2661) | <img src = 'assets/equations/GAN.png' height = '70px'>
**LSGAN**| [Arxiv](https://arxiv.org/abs/1611.04076) | <img src = 'assets/equations/LSGAN.png' height = '70px'>
**WGAN**| [Arxiv](https://arxiv.org/abs/1701.07875) | <img src = 'assets/equations/WGAN.png' height = '105px'>
**DRAGAN**| [Arxiv](https://arxiv.org/abs/1705.07215) | <img src = 'assets/equations/DRAGAN.png' height = '70px'>
**CGAN**| [Arxiv](https://arxiv.org/abs/1411.1784) | <img src = 'assets/equations/CGAN.png' height = '70px'>
**infoGAN**| [Arxiv](https://arxiv.org/abs/1606.03657) | <img src = 'assets/equations/infoGAN.png' height = '70px'>
**ACGAN**| [Arxiv](https://arxiv.org/abs/1610.09585) | <img src = 'assets/equations/ACGAN.png' height = '70px'>
**EBGAN**| [Arxiv](https://arxiv.org/abs/1609.03126) | <img src = 'assets/equations/EBGAN.png' height = '70px'>
**BEGAN**| [Arxiv](https://arxiv.org/abs/1702.08431) | <img src = 'assets/equations/BEGAN.png' height = '105px'>  

#### Variants of GAN structure
<img src = 'assets/etc/GAN_structure.png' height = '600px'>

### Some results for mnist
Network architecture of generator and discriminator is the exaclty sames as in [infoGAN paper](https://arxiv.org/abs/1606.0365).  
For fair comparison of core ideas in all gan variants, all implementations for network architecture are kept same except EBGAN and BEGAN. Small modification is made for EBGAN/BEGAN, since those adopt auto-encoder strucutre for discriminator. But I tried to keep the capacity of discirminator.

The following results can be reproduced with command:  
```
python main.py --dataset mnist --gan_type <TYPE> --epoch 25 --batch_size 64
```

#### Random generation
All results are randomly sampled.

*Name* | *Epoch 1* | *Epoch 2* | *Epoch 10*
:---: | :---: | :---: | :---: |
GAN | <img src = 'assets/mnist_results/random_generation/GAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/GAN_epoch001_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/GAN_epoch009_test_all_classes.png' height = '230px'>
LSGAN | <img src = 'assets/mnist_results/random_generation/LSGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/LSGAN_epoch001_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/LSGAN_epoch009_test_all_classes.png' height = '230px'>
WGAN | <img src = 'assets/mnist_results/random_generation/WGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/WGAN_epoch001_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/WGAN_epoch009_test_all_classes.png' height = '230px'>
DRAGAN | <img src = 'assets/mnist_results/random_generation/DRAGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/DRAGAN_epoch001_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/DRAGAN_epoch009_test_all_classes.png' height = '230px'>
EBGAN | <img src = 'assets/mnist_results/random_generation/EBGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/EBGAN_epoch001_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/EBGAN_epoch009_test_all_classes.png' height = '230px'>
BEGAN | <img src = 'assets/mnist_results/random_generation/BEGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/BEGAN_epoch001_test_all_classes.png' height = '230px'> | <img src = 'assets/mnist_results/random_generation/BEGAN_epoch009_test_all_classes.png' height = '230px'>

#### Conditional generation
Each row has the same noise vector and each column has the same label condition.

*Name* | *Epoch 1* | *Epoch 10* | *Epoch 25*
:---: | :---: | :---: | :---: |
CGAN | <img src = 'assets/mnist_results/conditional_generation/CGAN_epoch000_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/mnist_results/conditional_generation/CGAN_epoch009_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/mnist_results/conditional_generation/CGAN_epoch024_test_all_classes_style_by_style.png' height = '230px'>
ACGAN | <img src = 'assets/mnist_results/conditional_generation/ACGAN_epoch000_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/mnist_results/conditional_generation/ACGAN_epoch009_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/mnist_results/conditional_generation/ACGAN_epoch024_test_all_classes_style_by_style.png' height = '230px'>
infoGAN | <img src = 'assets/mnist_results/conditional_generation/infoGAN_epoch000_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/mnist_results/conditional_generation/infoGAN_epoch009_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/mnist_results/conditional_generation/infoGAN_epoch024_test_all_classes_style_by_style.png' height = '230px'>

#### InfoGAN : Manipulating two continous codes
<table align='center'>
<td><img src = 'assets/mnist_results/infogan/infoGAN_epoch024_test_class_c1c2_2.png' height = '200px'></td>
<td><img src = 'assets/mnist_results/infogan/infoGAN_epoch024_test_class_c1c2_5.png' height = '200px'></td>
<td><img src = 'assets/mnist_results/infogan/infoGAN_epoch024_test_class_c1c2_7.png' height = '200px'></td>
<td><img src = 'assets/mnist_results/infogan/infoGAN_epoch024_test_class_c1c2_9.png' height = '200px'></td>
</table>

### Some results for fashion-mnist
Comments on network architecture in mnist are also applied to here.  
[Fasion-mnist](https://github.com/zalandoresearch/fashion-mnist) is a recently proposed dataset consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

The following results can be reproduced with command:  
```
python main.py --dataset fashion-mnist --gan_type <TYPE> --epoch 40 --batch_size 100
```

#### Random generation
All results are randomly sampled.

*Name* | *Epoch 1* | *Epoch 20* | *Epoch 40*
:---: | :---: | :---: | :---: |
GAN | <img src = 'assets/fashion_mnist_results/random_generation/GAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/GAN_epoch019_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/GAN_epoch039_test_all_classes.png' height = '230px'>
LSGAN | <img src = 'assets/fashion_mnist_results/random_generation/LSGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/LSGAN_epoch019_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/LSGAN_epoch039_test_all_classes.png' height = '230px'>
WGAN | <img src = 'assets/fashion_mnist_results/random_generation/WGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/WGAN_epoch019_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/WGAN_epoch039_test_all_classes.png' height = '230px'>
DRAGAN | <img src = 'assets/fashion_mnist_results/random_generation/DRAGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/DRAGAN_epoch019_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/DRAGAN_epoch039_test_all_classes.png' height = '230px'>
EBGAN | <img src = 'assets/fashion_mnist_results/random_generation/EBGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/EBGAN_epoch019_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/EBGAN_epoch039_test_all_classes.png' height = '230px'>
BEGAN | <img src = 'assets/fashion_mnist_results/random_generation/BEGAN_epoch000_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/BEGAN_epoch019_test_all_classes.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/random_generation/BEGAN_epoch039_test_all_classes.png' height = '230px'>

#### Conditional generation
Each row has the same noise vector and each column has the same label condition.

*Name* | *Epoch 1* | *Epoch 20* | *Epoch 40*
:---: | :---: | :---: | :---: |
CGAN | <img src = 'assets/fashion_mnist_results/conditional_generation/CGAN_epoch000_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/conditional_generation/CGAN_epoch019_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/conditional_generation/CGAN_epoch039_test_all_classes_style_by_style.png' height = '230px'>
ACGAN | <img src = 'assets/fashion_mnist_results/conditional_generation/ACGAN_epoch000_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/conditional_generation/ACGAN_epoch019_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/conditional_generation/ACGAN_epoch039_test_all_classes_style_by_style.png' height = '230px'>
infoGAN | <img src = 'assets/fashion_mnist_results/conditional_generation/infoGAN_epoch000_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/conditional_generation/infoGAN_epoch019_test_all_classes_style_by_style.png' height = '230px'> | <img src = 'assets/fashion_mnist_results/conditional_generation/infoGAN_epoch039_test_all_classes_style_by_style.png' height = '230px'>

#### InfoGAN : Manipulating two continous codes
<table align='center'>
<td><img src = 'assets/fashion_mnist_results/infogan/infoGAN_epoch039_test_class_c1c2_1.png' height = '230px'></td>
<td><img src = 'assets/fashion_mnist_results/infogan/infoGAN_epoch039_test_class_c1c2_4.png' height = '230px'></td>
<td><img src = 'assets/fashion_mnist_results/infogan/infoGAN_epoch039_test_class_c1c2_5.png' height = '230px'></td>
<td><img src = 'assets/fashion_mnist_results/infogan/infoGAN_epoch039_test_class_c1c2_8.png' height = '230px'></td>
</table>

### Some results for celebA
(to be added)

## Variational Auto-Encoders (VAEs)

### Lists

*Name* | *Paper Link* |
:---: | :---: |
**VAE**| [Arxiv](https://arxiv.org/abs/1312.6114) 
**CVAE**| [Arxiv](https://arxiv.org/abs/1406.5298) 
**DVAE**| [Arxiv](https://arxiv.org/abs/1511.06406) 
**AAE**| [Arxiv](https://arxiv.org/abs/1511.05644) 

### Some results for mnist
(to be added)
### Some results for fashion-mnist
(to be added)
### Some results for celebA
(to be added)

## Acknowledgements
This implementation has been based on [this repository](https://github.com/carpedm20/DCGAN-tensorflow) and tested with Tensorflow over ver1.0 on Windows 10 and Ubuntu 14.04.
