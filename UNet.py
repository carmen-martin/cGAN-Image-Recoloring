import torch
import torch.nn as nn
import torch.nn.functional as F

# Double 3x3 convolution (convolution-BatchNorm-activation) 
class double_convolution(nn.Module):
    
    def __init__(self, in_channels, filters):
        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding='valid'),
            nn.BatchNorm2d(filters[0]),
            nn.ReLu(),
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding='valid'),
            nn.BatchNorm2d(filters[1]),
            nn.ReLu())
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.core_layers(x)


# Downsampling
class contracting_path(nn.Module):
    
    def __init__(self, in_channels, filters):
        self.downsample = nn.Sequential(
            double_convolution(in_channels, filters),
            nn.MaxPool2d(kernel_size=2, stride=2))
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.downsample(x)


# Upsampling
class expanding_path(nn.Module):
    
    def __init__(self, in_channels, filters, upsampling_mode='nearest'):
        # Options for unsampling_mode are : 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsampling_mode),
            nn.double_convolution(in_channels, filters))
        
        # Añadir concatenación con los features maps correspondientes de la extracting layer
            

# UNet