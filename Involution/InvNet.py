import torch
import torch.nn as nn
from Involution import Involution2D
from torchsummary import summary
from typing import Union, Tuple, Optional

class Bottleneck(nn.Module):
    
    def  __init__(self,
                  in_channels: int,
                  out_channels: int,
                  expansion: int = 4,
                  stride: Union[int, Tuple[int,int]] = (1,1),
                  dilation: Union[int, Tuple[int,int]] = (1,1),
                  downsample: Optional[nn.Module] = None,
                  ):
        """
        

        Parameters
        ----------
        in_channels : int
            Input Channels.
        out_channels : int
            Output Channels.
        expansion : int, optional
            The ratio of out_channels/mid_channels where
            mid_channels are the input/output channels of 
            conv2. The default is 4.
        stride : Union[int, Tuple[int,int]], optional
            Stride of the block. The default is (1,1).
        dilation : Union[int, Tuple[int,int]], optional
            Dilation of Convolution. The default is (1,1).
        downsample : Optional[nn.Module], optional
            Downsample operation on identity branch. The default is None.

        """
        
        super(Bottleneck, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = self.mid_channels,
                               kernel_size = (1,1),
                               stride = (1,1),
                               padding = (0,0), 
                               bias = False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        
        self.conv2 = Involution2D(channels = self.mid_channels,
                                  kernel_size = 7,
                                  stride = stride)
        
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        
        self.conv3 = nn.Conv2d(in_channels = self.mid_channels,
                               out_channels = out_channels,
                               kernel_size = 1,
                               bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        
    def forward(self, x):
        
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
    
class ResidualLayer(nn.Sequential):
    
    def __init__(self, 
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion = 4,
                 stride = 1,
                 avg_down = False,
                 **kwargs
                 ) -> nn.Sequential:
        """
        
        Parameters
        ----------
        block : int
            Residual block used to build ResidualLayer.
        num_blocks : int
            Number of blocks.
        in_channels : int
            Block's Input Channels.
        out_channels : int
            Block's Output Channels.
        expansion : int, optional
            The expansion for Bottleneck. The default is 4.
        stride : int, optional
            Stride of the first block. The default is 1.
        avg_down : bool
            Use Average pool instead of stride convolution. The default is False.

        Returns
        -------
        nn.Sequential (list of Residual layers)

        """
    
        self.block = block
        self.expansion = expansion
        
        downsample = None
        
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride!=1:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            
            downsample.extend([
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size = 1,
                          stride = conv_stride,
                          bias = False),
                nn.BatchNorm2d(out_channels)
                ])
            downsample = nn.Sequential(*downsample)
        
        layers = []
        
        layers.append(
            block(
                in_channels,
                out_channels,
                expansion = self.expansion, 
                stride = stride,
                downsample = downsample
                ))
        in_channels = out_channels
        
        for i in range(1, num_blocks):
            
            layers.append(
                block(
                    in_channels,
                    out_channels,
                    expansion = self.expansion,
                    stride = 1
                    )
                )
        super(ResidualLayer, self).__init__(*layers)
        
class InvNet(nn.Module):
    
    arch = {
        26: (Bottleneck, (1, 2, 4, 1)),
        38: (Bottleneck, (2, 3, 5, 2)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    
    def __init__(self,
                 depth,
                 num_classes,
                 in_channels = 3,
                 stem_channels = 64,
                 base_channels = 64,
                 expansion = 4,
                 num_stages = 4,
                 strides = (1,2,2,2),
                 dilations = (1,1,1,1),
                 out_indices = (3, ),
                 avg_down = False,
                 zero_init_residual = True
                 ):
        """
        

        Parameters
        ----------
        depth : int
            Network Depth {18, 34, 50, 101, 152}.
        num_classes : int
            Number of classes.
        in_channels : int, optional
            DESCRIPTION. The default is 3.
        stem_channels : int, optional
            Output channels for the stem layer. The default is 64.
        base_channels : TYPE, optional
            Middle Channels for the first stage. The default is 64.
        expansion : int, optional
            expansion for the bottleneck block. The default is 4.
        num_stages : int, optional
            Stages of the network. The default is 4.
        strides : tuple, optional
            Strides of the first blocks of each stage. The default is (1,2,2,2).
        dilations : tuple, optional
            Dilation of each stage. The default is (1,1,1,1).
        out_indices : tuple, optional
            Output from which stages, if only one stage is specified,
            a single tensor(feature map) is returned, 
            otherwise a tuple of tensors will be returned. The default is (3, ).
        avg_down : bool, optional
            Use Average pool instead of strided convolution when downsampling 
            in bottleneck. The default is False.
        zero_init_residual : bool, optional
            Whether to use zero init for last batchnorm layer
            in res blocks or not. The default is True.

        """
        
        super(InvNet, self).__init__()
        if depth not in self.arch:
            raise KeyError(f'Invalid depth {depth} for InvNet.')
        
        self.num_classes = num_classes
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >=1 and num_stages <=4
        self.strides = strides
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.avg_down = avg_down
        self.zero_init_residual = zero_init_residual
        self.expansion = 4
        assert self.expansion != 0
        self.block, stage_blocks = self.arch[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(512 * self.expansion, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        self.stem_layer(in_channels, stem_channels)
                
        self.res_layers = []
        in_c = stem_channels
        out_c = base_channels * self.expansion
        
        for i, num_blocks in enumerate(self.stage_blocks):
            
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                        block = self.block,
                        num_blocks = num_blocks,
                        in_channels = in_c,
                        out_channels = out_c,
                        expansion = self.expansion,
                        stride = stride,
                        dilation = dilation,
                        avg_down = self.avg_down,
                        )
            in_c = out_c
            out_c *= 2
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            
    def make_res_layer(self, **kwargs):
        return ResidualLayer(**kwargs)
    
    def stem_layer(self, in_channels, stem_channels):
        
        self.stem = nn.Sequential(nn.Conv2d(in_channels,
                                            stem_channels // 2,
                                            kernel_size = 3,
                                            stride = 2, 
                                            padding = 1),
                                  nn.BatchNorm2d(stem_channels // 2),
                                  nn.ReLU(inplace=True),
                                  Involution2D(stem_channels // 2, 
                                               kernel_size = 3,
                                               stride = 1),
                                  nn.BatchNorm2d(stem_channels // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(stem_channels // 2,
                                            stem_channels,
                                            kernel_size =3,
                                            stride = 1,
                                            padding = 1),
                                  nn.BatchNorm2d(stem_channels),
                                  nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
    def init_weights(self, pretrained = None):
            
            super(InvNet, self).init_weights(pretrained)
            if pretrained is None:
                for i in self.modules():
                    if isinstance(i, nn.Conv2d):
                        nn.init.kaiming_normal_(i, mode = 'fan_out', nonlinearity='relu')
                    elif isinstance(i, nn.BatchNorm2d):
                        nn.init.constant_(i, 1.0)
                
                if self.zero_init_residual:
                    for i in self.modules():
                        if isinstance(i, Bottleneck):
                            nn.init.constant_(i.bn3, 0.0)
        
    def forward(self, x):
            
            x = self.stem(x)
            x = self.maxpool(x)
            outs = []
            for i, layer_name in enumerate(self.res_layers):
                
                res_layer = getattr(self, layer_name)
                x = res_layer(x)
                if i in self.out_indices:
                    outs.append(x)
            
            if len(outs) == 1:
                outs = outs[0]
            else:
                outs = tuple(outs)
                
            outs = self.avgpool(outs)
            outs = torch.flatten(outs, 1)
            outs = self.fc1(outs)
            outs = self.dropout(outs)
            outs = self.fc2(outs)
            
            return outs
        
        
def test():
    
    x = torch.randn((1,3,64,64)).to(torch.device('cuda'))
    model = InvNet(18, num_classes=3).to(torch.device('cuda'))
    output = model(x)
    print(summary(model, (3,64,64)))
    print(output.shape)
    
if __name__ == '__main__':
    
    test()       
