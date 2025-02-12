# Nested U-Net (U-Net++) for Car Segmentation

This project implements the **Nested U-Net (U-Net++)** architecture for **image segmentation**, specifically focusing on car segmentation. U-Net++ improves upon the original U-Net by introducing **nested skip pathways**, enhancing the flow of information between the encoder and decoder. The implementation is built using **PyTorch** and leverages data from the [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge/overview).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Model](#model)
- [Training and Testing](#training-and-testing)
- [Metrics](#metrics)
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

## Model
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
```python
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, first_feature=32, L=4, deep_supervision=True, device='cpu'):
        super(UNET, self).__init__()

        # Ensure the proper parameters are given
        assert isinstance(in_channels, int), "in_channels must be an integer"
        assert isinstance(out_channels, int), "out_channels must be an integer"
        assert isinstance(first_feature, int), "first_feature must be an integer"
        assert isinstance(L, int), "L must be an integer"
        assert L <= 4 and L > 0, "L must be in range [1, 5)"
        assert isinstance(deep_supervision, bool), "deep_supervision must be True or False"

        # L is used in defining how deep the network is. Essentially, there will be L + 1 layers (top to bottom of triangular architecture)
        # with each layer having one less cell than the last. Paper uses L in range [1, 4]
        self.L = L

        # Device to send tensors to
        self.device = device

        # Deep supervision will define if the model backward propogates using all cells of the top layer (excluding backbone)
        self.deep_supervision = deep_supervision

        # Feature dimensionality of the model (paper starts with 32 and increases by 2 each layer)
        features = [first_feature * 2 ** x for x in range(0, L + 1, 1)]

        # Backbone modules and pooling
        self.backbone = nn.ModuleList([])
        self.pool = nn.MaxPool2d(2, 2)

        # Create dense convolutional blocks as backbone
        for feature in features:
            self.backbone.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Create 2-dimensional list of nested dense convolutional blocks
        self.layers = nn.ModuleList([
            nn.ModuleList([ConvBlock(features[i] * 1 + features[i+1], features[i]) for i in range(L)]),
            nn.ModuleList([ConvBlock(features[i] * 2 + features[i+1], features[i]) for i in range(L - 1)]),
            nn.ModuleList([ConvBlock(features[i] * 3 + features[i+1], features[i]) for i in range(L - 2)]),
            nn.ModuleList([ConvBlock(features[i] * 4 + features[i+1], features[i]) for i in range(L - 3)]),
        ])[:L] # Only use the first L layers

        # Upsampling used in paper
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Output module depending on deepsupervision
        if deep_supervision:
            self.deep_layers = nn.ModuleList([
                nn.Conv2d(features[0], out_channels, kernel_size=1) for _ in range(L)
            ])
        else:
            self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)
```
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

## Training and Testing
### Training Step
The `train_step` function trains the model on a batch of data using a loss function and optimizer.
```python
    def train_step(self, loader: DataLoader, criterion, optimizer: torch.optim.Adam, epoch: int, num_epochs: int, miniters: int=10):
        self.train()
        total_loss = 0
        loop = tqdm(loader, total=len(loader), desc=f"[{epoch+1}/{num_epochs}] loss=inf", miniters=miniters)
        for idx, (x_batch, y_batch) in enumerate(loop):
            x_batch: torch.Tensor
            y_batch: torch.Tensor
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            optimizer.zero_grad()

            if self.deep_supervision:
                logits = self(x_batch)
                loss = 0
                for logit in logits:
                    loss += criterion(logit, y_batch)
                loss /= len(logits)
            else:
                logits = self(x_batch)
                logits = logits.squeeze(1)
                loss = criterion(logits, y_batch.squeeze(1))
            
            total_loss += loss.item()
            
            if (idx + 1) % miniters == 0:
                loop.desc = f"[{epoch+1}/{num_epochs}] loss={(total_loss / (idx + 1)):.4f}"

            loss.backward()
            optimizer.step()

        return total_loss / len(loader)
```
### Evaluation Step
The `evaluate` function computes the loss over the validation set without updating the model
```python
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, criterion):
        self.eval()
        total_loss = 0

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            if self.deep_supervision:
                logits = self(x_batch)
                loss = sum(criterion(logit, y_batch) for logit in logits) / len(logits)
            else:
                logits = self(x_batch).squeeze(1)
                loss = criterion(logits, y_batch.squeeze(1))

            total_loss += loss.item()

        return total_loss / len(loader)
```

## Metrics
### IoU Score
Intersection over Union (IoU) measures segmentation accuracy.
```python
    def _iou_calculation(self, outputs: torch.Tensor, labels: torch.Tensor, smooth: float = 1e-6):
        outputs = (outputs > 0.5).float() #  Convert to boolean mask
        labels = (labels > 0.5).float() #  Convert to boolean mask
        
        intersection = (outputs * labels).sum() #  Element-wise AND
        union = (outputs + labels).clamp(0, 1).sum() #  Element-wise OR

        iou = intersection / (union + smooth) #  Smooth to avoid division by zero
        
        return iou.cpu().item() #  Return IoU values
    
    @torch.no_grad()
    def iou_score(self, x: torch.Tensor, y: torch.Tensor):
        self.eval()

        x, y = x.to(self.device), y.to(self.device)
        x = x.unsqueeze(0)

        preds = self(x)[-1] if self.deep_supervision else self(x)

        preds = torch.sigmoid(preds)

        preds = (preds > 0.5).float()
        preds.squeeze_(1)

        return self._iou_calculation(preds, y.squeeze(1))
```
### Dice Score
The Dice coefficient is another metric for evaluating segmentation performance.
```python
    @torch.no_grad()
    def dice_score(self, x: torch.Tensor, y: torch.Tensor):
        self.eval()  # Set model to evaluation mode

        assert 4 >= x.dim() > 2, f"Input dimensions must be in range (2, 4], got {x.dim()}"

        x, y = x.to(self.device), y.to(self.device)

        if x.dim() == 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        preds = self(x)[-1] if self.deep_supervision else self(x)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float().squeeze(1) # ,Convert to binary mask

        # Compute Dice Score
        dice_score = (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        return dice_score.item()
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

