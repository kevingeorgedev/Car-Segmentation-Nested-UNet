# Nested U-Net (U-Net++) for Car Segmentation

This project implements the **Nested U-Net (U-Net++)** architecture for **image segmentation**, specifically focusing on car segmentation. U-Net++ improves upon the original U-Net by introducing **nested skip pathways**, enhancing the flow of information between the encoder and decoder. The implementation is built using **PyTorch** and leverages data from the [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge/overview).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Evaluation](#evaluation)
- [Results](#results)

## Introduction
U-Net++ is a convolutional neural network (CNN) designed for **semantic segmentation**. Unlike traditional U-Net, U-Net++ incorporates **dense skip connections**, which improve gradient flow and feature reuse. This enables the model to learn fine-grained details more effectively, making it particularly useful for **car segmentation** and other image segmentation tasks.

## Features
- **Nested Skip Connections**: Enhances feature fusion between encoder and decoder.
- **Efficient Encoder-Decoder Structure**: Retains spatial information while learning hierarchical features.
- **Data Augmentation**: Improves generalization by applying transformations to training images.
- **PyTorch Implementation**: Fully compatible with GPU acceleration for faster training.

## Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/kevingeorgedev/Car-Segmentation-Nested-UNet.git
cd unet-image-segmentation
pip install -r requirements.txt
```

Additionally, install **PyTorch** following the official guide: [PyTorch Installation](https://pytorch.org/get-started/locally/).

## U-Net Model with Dense Convolutional Blocks
### Overview
This implementation of Nested U-Net (U-Net++) is designed for image segmentation tasks, featuring dense convolutional blocks, nested skip pathways, and deep supervision. The network can handle varying depth levels (L) and offers configurable options for output supervision.

### Model Components
#### 1. `ConvBlock` (Dense Convolutional Block: $X^{i, j} \text{ Convolution}$)
This is the basic building block of the Nested U-Net, consisting of two convolutional layers followed by batch normalization and ReLU activation.
```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
```
#### 2. `UNET` (Nested U-Net Model)
The U-Net model consists of multiple levels (L), where each level includes convolutional blocks and skip connections. It allows for deep supervision, meaning multiple outputs at different levels can be used to guide training.
##### **Key Features:**
- **Backbone:** The contracting path extracts feature maps using convolutional blocks.
- **Dense Nested Layers:** Multiple convolutional blocks are used per level.
- **Upsampling:** Uses bilinear interpolation to scale up feature maps.
- **Deep Supervision:** If enabled, multiple outputs are used for loss calculation.
##### **Initialization Parameters:**
- **in_channels:** Number of input channels.
- **out_channels:** Number of output channels.
- **first_feature:** Initial feature size (doubles per layer).
- **L:** Depth of the network (maximum 4).
- **deep_supervision:** Boolean flag for deep supervision.
- **device:** Defines the computing device.
##### **Forward Pass**

It looks like you want to keep everything under the **2. UNET** section while ensuring clarity and conciseness. Here’s a refined version that keeps all details while reducing redundancy:

---

### **2. UNET (Nested U-Net Model)**  
The U-Net model consists of **L** levels, each with convolutional blocks and skip connections. It supports **deep supervision**, allowing multiple outputs at different levels to guide training.  

#### **Key Features:**  
- **Backbone:** Contracting path extracts feature maps via convolutional blocks.  
- **Dense Nested Layers:** Multiple convolutional blocks per level.  
- **Upsampling:** Bilinear interpolation for feature map scaling.  
- **Deep Supervision (Optional):** Uses multiple outputs for loss calculation.  

#### **Initialization Parameters:**  
- **in_channels, out_channels:** Input/output channel sizes.  
- **first_feature:** Initial feature size (doubles per layer).  
- **L:** Network depth (max 4).  
- **deep_supervision:** Enables multi-output loss calculation.  
- **device:** Specifies computing device.  

---

This keeps all the information while making it more structured and compact. Let me know if you need further refinements! 🚀

## Evaluation
The model is evaluated on standard segmentation metrics:

| Metric     | Value  |
|------------|--------|
| IoU        | 0.989  |
| Dice Score | 0.994  |
| Accuracy   | 99.8%  |

![IoU vs Epochs](metric_tests.png "IoU vs Epochs")

## Results
### Predicted Mask vs. Ground Truth

Side-by-side comparison of a test image and its predicted mask:

![Image vs Predicted Mask](image_vs_pred.png "Image vs Predicted Mask")

### Mask Applied Over Test Image

Overlaying the predicted mask on the original image:

![Applied Mask](AppliedMask.png "Applied Mask")

---
**Contributions & Feedback**
Feel free to submit issues, feature requests, or contribute to the repository!

