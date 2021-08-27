# Image Forensics

CS 754 Advanced Image Processing Course Project
Guide: Prof. Ajit Rajwade

This repository consists of code to distinguish between photographic images, photorealistic computer graphics and GAN generated images. Two different approaches were experimented with, one trained an SVM classifier on features extracted from wavelet decomposition of RGB color bands and other trained a deep convolutional neural networks like CaffeNet and ResNet-50 on high pass filtered images. Accuracy achieved was more than 95% for all 3 classes in both the methods.

The photographic images and computer graphics were obtained from [Columbia Photographic Images and Photorealistic Computer Graphics Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/PIM_PRCG_dataset/) and the GAN generated images from taken from results of [Nvidia StyleGAN](https://github.com/NVlabs/stylegan).

Here is the [link](https://drive.google.com/drive/folders/1PaYpCYJBpkkLKvf6Cu1Dqjxd_T97lCSl?usp=sharing) to the complete dataset of images used in our experiment.

### References 

1. [Identifying Materials of Photographic Images and Photorealistic Computer Generated Graphics Based on Deep CNNs](https://www.researchgate.net/profile/Qi-Cui-6/publication/325699635_Identifying_materials_of_photographic_images_and_photorealistic_computer_generated_graphics_based_on_deep_CNNs/links/5cac14d392851c64bd59eb75/Identifying-materials-of-photographic-images-and-photorealistic-computer-generated-graphics-based-on-deep-CNNs.pdf)
2. [How Realistic is Photorealistic?](https://www.researchgate.net/publication/3319160_How_Realistic_is_Photorealistic)
3. [Distinguishing computer graphics from natural images using convolution neural networks](https://ieeexplore.ieee.org/abstract/document/8267647?casa_token=9VkzmQUJDMoAAAAA:TqrVEZnCavB1Z9ZyoSo-m0wnu9wA6Q0AKvVg9lsAta2ZVkv5zQz8y0SAg0efg9igO-x0y04QCTPs)
4. [Identification of Deep Network Generated Images Using Disparities in Color Components](https://arxiv.org/abs/1808.07276)
5. [Deep Learning for Deepfakes Creation and Detection](https://arxiv.org/pdf/1909.11573v1.pdf)
6. [Deepfake detection(repository)](https://github.com/HongguLiu/Deepfake-Detection)
7. [Higher-order Wavelet Statistics and their Application to Digital Forensics](https://www.researchgate.net/publication/4374571_Higher-order_Wavelet_Statistics_and_their_Application_to_Digital_Forensics)
8. [Natural Image Statistics in Digital Image Forensics*](https://www.researchgate.net/publication/239545646_Natural_Image_Statistics_in_Digital_Image_Forensics)
