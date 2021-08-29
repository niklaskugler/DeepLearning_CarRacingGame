# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:02:21 2021

@author: Marten Kreis
"""

import torch
import torch.nn as nn

import glob
from PIL import Image
from torchvision import transforms


class DenseLayer(nn.Module):  # also called bottleneck Layer (Used for DenseNet-BC Variants)
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
        # 1x1 kernel hat ko + k*(l-1) input channels, und erzeugt 4*32 outputs, welche zu 3x3 kernel gehen. Dies dient vor allem der Parameterreduzierung
        self.bn2 = nn.BatchNorm2d(out_channels * 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # DenseLayer defined as mentioned in paper (picture: Architecture.png)

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))  # input wird zunächst von 1x1 kernel gefiltert
        out = self.conv2(
            self.relu2(self.bn2(out1)))  # anschließend wird 3x3 kernel genutzt, welcher 32 output channels erzeugt
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x))))
        return out


class DenseBlock(nn.Module):
    def __init__(self, layer_size, in_channels,
                 growth_rate):  # growth_rate (k in paper) is equal to amount of output channels for each layer
        super(DenseBlock, self).__init__()

        self.block = []
        for i in range(layer_size):
            self.block.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
            # input = k0 + (l-1)*k, k0 = in_channels, l-1 = i (l = amount of layers before current layer)

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        out = self.block(x)
        return out


class DenseNet(nn.Module):
    def __init__(self, in_channels=3, layer_size=None, growth_rate=12,
                 additional_neurons=2):
        # in_channels describes picture channels (3 for rgb)
        # layer_size must be list of 4 Elements for 4 DenseBlocks
        # ToDo: Adapt growth rate here
        # ToDo: Adapt in_channels according to input image
        super(DenseNet, self).__init__()

        if layer_size is None:
            layer_size = [3, 7, 10, 7]
            # layer_size = [7, 7, 7, 6]
            # Todo: Adapt amount of layers in each Denseblock here, try both or more variants of layers out
        self.out_channels = 2 * growth_rate  # for k = 32 ->  64
        self.in_channels = 0  # used later

        # First Convolution & Pooling
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # FirstDenseBlock
        self.in_channels = self.out_channels  # 2*growth_rate -> 64
        print("Channels in DenseBlock1: ", self.in_channels)
        self.out_channels = growth_rate
        self.DenseBlock1 = DenseBlock(layer_size[0], self.in_channels, growth_rate)  # 6 Denselayer

        # First Transition Layer
        self.in_channels = self.in_channels + growth_rate * layer_size[0]  # 256
        print("Channels in Transition Layer 1: ", self.in_channels)
        self.out_channels = int(self.in_channels / 2)  # 128
        self.TransitionLayer1 = TransitionBlock(self.in_channels, self.out_channels)

        # Second DenseBlock
        self.in_channels = self.out_channels  # 128 in channels
        print("Channels in DenseBlock2: ", self.in_channels)
        self.out_channels = growth_rate
        self.DenseBlock2 = DenseBlock(layer_size[1], self.in_channels, growth_rate)  # 12 Denselayer

        # Second Transition Layer
        self.in_channels = self.in_channels + growth_rate * layer_size[1]  # 512
        print("Channels in Transition Layer 2: ", self.in_channels)
        self.out_channels = int(self.in_channels / 2)  # 256
        self.TransitionLayer2 = TransitionBlock(self.in_channels, self.out_channels)

        # Third DenseBlock
        self.in_channels = self.out_channels  # 256 in channels
        print("Channels in DenseBlock3: ", self.in_channels)
        self.out_channels = growth_rate
        self.DenseBlock3 = DenseBlock(layer_size[2], self.in_channels, growth_rate)  # 24 Denselayer

        # Third Transition Layer
        self.in_channels = self.in_channels + growth_rate * layer_size[2]  # 1024
        print("Channels in Transition Layer 3: ", self.in_channels)
        self.out_channels = int(self.in_channels / 2)  # 512
        self.TransitionLayer3 = TransitionBlock(self.in_channels, self.out_channels)

        # Fourth DenseBlock
        self.in_channels = self.out_channels  # 512 in channels
        print("Channels in DenseBlock4: ", self.in_channels)
        self.out_channels = growth_rate
        self.DenseBlock4 = DenseBlock(layer_size[3], self.in_channels, growth_rate)  # 16 Denselayer

        # Global Average Pooling -> Compresses all channels of size 3x2 to size 1x1 (for input neurons)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=(5, 6), stride=1,
                                            padding=0)  # kernel size of 3x2 depends on input size of image!!!!

        self.in_channels = self.in_channels + growth_rate * layer_size[3]  # 1024
        self.batchnorm2 = nn.BatchNorm2d(self.in_channels)

        # fully connected layer
        self.in_channels = self.in_channels + additional_neurons
        print("Channels in In Fully Connected: ", self.in_channels)
        self.fully_connected = nn.Linear(self.in_channels, 4)
        # in_features = 1024 + additional neurons used as input, output = 3 (steering, acceleration, brake)

        self.sigmoid = nn.Sigmoid()

        # initialization of all weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, list_additional_inputs=[1,
                                                 0.5]):  # x = Tensor of Image (96*85*in_channels -> specified in constructor of DenseNet), a = List of other Inputs
        print("Size before First Convolution & Maxpooling:", x.shape)
        x = self.maxpool(self.relu1(self.batchnorm1(self.conv1(x))))
        print("Size before DenseBlock1:", x.shape)
        x = self.TransitionLayer1(self.DenseBlock1(x))
        print("Size before DenseBlock2:", x.shape)
        x = self.TransitionLayer2(self.DenseBlock2(x))
        print("Size before DenseBlock3:", x.shape)
        x = self.TransitionLayer3(self.DenseBlock3(x))
        print("Size before DenseBlock4:", x.shape)
        x = self.DenseBlock4(x)
        print("Size before Global Average Pooling:", x.shape)
        x = self.global_avg_pool(x)
        x = self.batchnorm2(x)
        print("Size before flattening:", x.shape)
        x = torch.flatten(x, 1)     # Put Output in one dimensional tensor (vector)
        print("Size before adding aditional input neurons:", x.shape)
        # Putting Output in 1-dimensional matrix and adding adittional input neurons
        list_additional_inputs = torch.Tensor(list_additional_inputs)
        list_additional_inputs = torch.unsqueeze(list_additional_inputs, 0)  # unsqueeze tensor, so it can be concatenated with tensor x
        x = torch.cat((x, list_additional_inputs), 1)  # concatenate tensor x and list_additional_inputs -> new tensor of size([1,1026])
        print("Size before FullyConnectedLayer:", x.shape)
        output = self.fully_connected(x)
        output = self.sigmoid(output)
        return output


def count_parameters(model):  # Function to count learnable parameters of model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


CNN = DenseNet()
# print(CNN)

image = glob.glob("C:\\Users\\Jakob Häringer\\Documents\\01 Studium\\08 6. Semester\\06 Software-Projekt\\"
                  "Eigene_Implementations\\DenseNet\\processed_1619865125_220.png")
img = Image.open(image[0])
img = transforms.ToTensor()(img).unsqueeze_(0)  # size(1,3,96,84)
additional_input_neurons = [3, 0]  # true speed, steering angle
CNN.eval()  # activate evaluation mode
out = CNN(img, additional_input_neurons)
print("Output of CNN:", out)  # pass sample image to CNN and print output
print("Output Size of CNN", out.shape)
print("Amount of learnable Parameters: ", count_parameters(CNN))
