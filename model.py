from email.policy import strict
from gzip import _PaddedFile
from posixpath import split
from tracemalloc import start
from turtle import forward
import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_features)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YoloV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YoloV1, self).__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers()
        self.fcs = self._create_fcs(**kwargs)
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self):
        layers = nn.Sequential(
            CNNBlock(in_features=self.in_channels, out_features=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(in_features=64, out_features=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            CNNBlock(in_features=192, out_features=128, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_features=128, out_features=256, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_features=256, out_features=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_features=256, out_features=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            CNNBlock(in_features=512, out_features=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_features=256, out_features=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_features=512, out_features=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_features=256, out_features=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_features=512, out_features=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_features=256, out_features=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_features=512, out_features=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_features=256, out_features=512, kernel_size=3, stride=1, padding=1),
            
            CNNBlock(in_features=512, out_features=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_features=512, out_features=1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            CNNBlock(in_features=1024, out_features=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_features=512, out_features=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_features=1024, out_features=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_features=512, out_features=1024, kernel_size=3, stride=1, padding=1),
            
            CNNBlock(in_features=1024, out_features=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_features=1024, out_features=1024, kernel_size=3, stride=2, padding=1),
            CNNBlock(in_features=1024, out_features=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_features=1024, out_features=1024, kernel_size=3, stride=1, padding=1),
        )
        
        return layers

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024*S*S, out_features=496),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=496, out_features=S*S*(C+B*5))
        )


# model = YoloV1(in_channels=3, split_size=7, num_boxes=2, num_classes=20)
# print(model)