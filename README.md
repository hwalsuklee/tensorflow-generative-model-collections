# tensorflow-generative-model-collections
Tensorflow implementation of various GANs and VAEs.

## Generative Adversarial Networks (GANs)
### Lists  

*Name* | *mnist impl.* | *fashion-mnist impl.* | *celebA impl.* | Loss Function
:---: | :---: | :---: | :---: | :--- |
[**GAN**](https://arxiv.org/abs/1406.2661) | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: | <img src = 'assets/equations/GAN.png' height = '60px'>
[**LSGAN**](https://arxiv.org/abs/1611.04076) | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: | <img src = 'assets/equations/LSGAN.png' height = '60px'>
[**WGAN**](https://arxiv.org/abs/1701.07875) | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: | <img src = 'assets/equations/WGAN.png' height = '90px'>
[**DRAGAN**](https://arxiv.org/abs/1705.07215) | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: | <img src = 'assets/equations/DRAGAN.png' height = '60px'>
[**CGAN**](https://arxiv.org/abs/1411.1784) | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: | <img src = 'assets/equations/CGAN.png' height = '60px'>
[**infoGAN**](https://arxiv.org/abs/1606.03657) | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: | <img src = 'assets/equations/infoGAN.png' height = '60px'>
[**ACGAN**](https://arxiv.org/abs/1610.09585) | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: | <img src = 'assets/equations/ACGAN.png' height = '60px'>
[**EBGAN**](https://arxiv.org/abs/1609.03126) | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: | <img src = 'assets/equations/EBGAN.png' height = '60px'>
[**BEGAN**](https://arxiv.org/abs/1702.08431) | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: | <img src = 'assets/equations/BEGAN.png' height = '90px'>  

:white_check_mark: means that it is implemented.  
:clock9: means that I'm working on it.  
:negative_squared_cross_mark: means that it will be implmented very soon.

### Results for mnist


## Variational Auto-Encoders (VAEs)

### Lists

*Name* | *mnist impl.* | *fashion-mnist impl.* | *celebA impl.*
:---: | :---: | :---: | :---:
[**VAE**](https://arxiv.org/abs/1312.6114) | :clock9: | :negative_squared_cross_mark: | :negative_squared_cross_mark:
[**CVAE**](https://arxiv.org/abs/1406.5298) | :clock9: | :negative_squared_cross_mark: | :negative_squared_cross_mark:
[**DVAE**](https://arxiv.org/abs/1511.06406) | :clock9: | :negative_squared_cross_mark: | :negative_squared_cross_mark:
[**AAE**](https://arxiv.org/abs/1511.05644) | :clock9: | :negative_squared_cross_mark: | :negative_squared_cross_mark:  

:white_check_mark: means that it is implemented.  
:clock9: means that I'm working on it.  
:negative_squared_cross_mark: means that it will be implmented very soon.  

## Acknowledgements
This implementation has been tested with Tensorflow over ver1.0 on Windows 10 and Ubuntu 14.04.
