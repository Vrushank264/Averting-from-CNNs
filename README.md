# Averting-from-CNNs
Code for our Research paper(Under Review) project called "Averting from CNNs for medical image classification".

## Abstract

This paper attempts to apply newer approaches that do not use CNNs conventionally to the evolving field of medical image classification. While analyzing, firstly, an all MLP architecture MLP-Mixer and secondly, the inverted convolutional kernels coined as Involution with our baseline ResNets, both models yield comparable results in detecting Covid19 and pneumonia using Chest X-ray images. On top of that, merging Involution kernels into ResNet architectures can produce promising performance while training on roughly 40% fewer parameters. This paper further compares these two architectures with various CNN-based models. We hope this research further helps the research community to utilize the capabilities of these newly introduced architectures in the medical field.
 
## Result:

![graphs](https://github.com/Vrushank264/Averting-from-CNNs/blob/main/graphs.png)

(Note: Vgg19 is treated as an outlier due to its high parameter size.)
