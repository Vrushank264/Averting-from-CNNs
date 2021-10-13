import torch
import torch.nn as nn
from typing import Tuple,Union

class Involution2D(nn.Module):

    def __init__(self,
                 channels: int,
                 kernel_size: Union[int, Tuple[int,int]],
                 stride: Union[int, Tuple[int,int]],
                 reduction_ratio: int = 4
                 ) -> torch.Tensor:
        """
        

        Parameters
        ----------
        channels : int
            number of input channels.
        kernel_size : Union[int, Tuple[int,int]]
            kernel size of Involution.
        stride : Union[int, Tuple[int,int]]
            stride.
        reduction_ratio : int, optional
            to control the intermediate channel dimensions,
            also helps in increasing the efficiency. The default is 4.

        Returns
        -------
        None
            A 4D Tensor.(batch_size, channels, height, width)

        """
    
        super(Involution2D, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio
        assert reduction_ratio != 0 and type(reduction_ratio) == int 
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        
        self.conv1 = nn.Conv2d(in_channels = channels,
                               out_channels = channels // reduction_ratio,
                               kernel_size = 1, 
                               stride = 1, 
                               padding=0
                               )
        self.bn1 = nn.BatchNorm2d(channels // reduction_ratio)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels = channels // reduction_ratio,
                               out_channels = kernel_size**2 * self.groups,
                               kernel_size= 1,
                               stride = 1, 
                               padding = 0
                               )
        if stride > 1:
            
            self.avgpool = nn.AvgPool2d(stride, stride)
        
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1) // 2, stride)
        
    def forward(self, x):
        
        weight = self.conv1(x if self.stride == 1 else self.avgpool(x))
        weight = self.bn1(weight)
        weight = self.relu(weight)
        weight = self.conv2(weight)
        b, c, h, w = weight.shape
        
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim = 3).view(b,self.channels, h, w)
        
        return out
