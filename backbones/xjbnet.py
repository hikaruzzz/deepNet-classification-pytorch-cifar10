import torch
import torch.nn as nn
import numpy as np


class Unit(nn.Module):
    '''把 conv + bn + relu层打包为一个unit，给后面的Sequential串联'''

    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels,stride=1,padding=1)
        # 注意此处，padding, 若=0，则input=32x32情况下，多层之后就会出现kernel_size > input_size的bug
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class XJBNet(nn.Module):
    def __init__(self,class_n=10):
        super(XJBNet,self).__init__()

        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.unit2 = Unit(in_channels=32,out_channels=32)
        self.unit3 = Unit(in_channels=32,out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)  # pool层


        self.unit4 = Unit(in_channels=32,out_channels=64)
        self.unit5 = Unit(in_channels=64,out_channels=64)
        self.unit6 = Unit(in_channels=64,out_channels=64)
        self.unit7 = Unit(in_channels=64,out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64,out_channels=128)
        self.unit9 = Unit(in_channels=128,out_channels=128)
        self.unit10 = Unit(in_channels=128,out_channels=128)
        self.unit11 = Unit(in_channels=128,out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128,out_channels=128)
        self.unit13 = Unit(in_channels=128,out_channels=128)
        self.unit14 = Unit(in_channels=128,out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # connect all units sequentially
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc1 = nn.Linear(in_features=128,out_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=class_n)


    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,128)
        output = self.fc1(output)
        output = self.fc2(output)

        return output

