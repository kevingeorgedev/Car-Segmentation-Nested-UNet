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
```python
    def forward(self, x):
        # Skip connections array, updates to skip_connections will eventually form 
        # what looks like an upper triangular matrix the is flipped horizontally
        skip_connections = []

        # Create the backbone
        back_bone_skip = []
        for idx, block in enumerate(self.backbone): #  Iterate backbone layers
            x = block(x)             #  Forward propagation for each layer of backbone
            back_bone_skip.append(x) #  Append skip connection to backbone
            if idx != self.L:        #  Not necessary to pool for last layer of backbone
                x = self.pool(x)     #  Max pool for next layer

        # Append backbone layer to skip connections and delete back_bone skip from memory
        # because it won't be used anymore
        skip_connections.append(back_bone_skip)
        del back_bone_skip

        # Iterate throught all nested layers
        for i in range(len(self.layers)):
            layer_skips = []                                        #  Array that holds the output skips for each cell in layer
            for l, layer in enumerate(self.layers[i]):              #  Enumerate the nested cells in layer
                cat_connections = []                                #  Connections to concatenate after forward propagation
                for k in range(i + 1):                              #  Iterate all previous cells in layer besides backbone
                    cat_connections.append(skip_connections[k][l])  #  Append nested cell to concatenate later
                cat_connections.append(self.up(skip_connections[i][l+1]))
                cat_connection = torch.cat(cat_connections, dim=1)  #  Finally, concatenate all cells to propagate through the next
                layer_skip = layer(cat_connection)                  #  Forward propagation for next cell
                layer_skips.append(layer_skip)                      #  Append new cell forward propagation to layer skips

            skip_connections.append(layer_skips)                    # Add layer skips to skip connections

        # We only want the output of each cell in the top row of architecture (besides backbone)
        # If not using deep supervision, use the output of cell x_(0, L)
        if self.deep_supervision: 
            return [self.deep_layers[x](skip_connections[x+1][0]) for x in range(self.L)]
        else: 
            return self.output(skip_connections[-1][-1])
```

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

