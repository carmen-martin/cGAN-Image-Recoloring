import torch
from torch import cat
import torch.nn as nn
from torchvision.transforms.functional import crop

# Double 3x3 convolution (convolution-BatchNorm-activation) 
class double_convolution(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.core_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='valid'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='valid'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
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


# Downsampling (doubleconv+pooling)
class downsampling(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            double_convolution(in_channels, out_channels)
        )
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
        return self.downsample(x)


# Upsampling (doubleconv+upsampling)
class upsampling(nn.Module):
    
    def __init__(self, in_channels, out_channels, upsampling_mode):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsampling_mode), #mode can be:'nearest','linear','bilinear','bicubic','trilinear'
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        )
        self.double_conv = double_convolution(in_channels, out_channels)

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

    def forward(self, x, x_down):   
        x = self.upsample(x)
        
        # Crop and concat corresponding feature map
        H = x.size(dim=2)
        W = x.size(dim=3)  
        top = (x_down.size(dim=2) - H)//2
        left = (x_down.size(dim=3) - W)//2
        
        cropped_x_down = crop(x_down, top, left, H, W)
        x = cat((x, cropped_x_down), dim=1) #check dim
        
        return self.double_conv(x)
    
# Final layer (class mapping)
class final(nn.Module):
    
    def __init__(self, in_channels, n_outputs):
        super().__init__()
        self.output = nn.Conv2d(in_channels, n_outputs, kernel_size=1)
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
        return self.output(x) 

# UNet
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, upsamplig_mode='nearest'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Input process
        self.input_process = double_convolution(in_channels, 64)
        
        # Contracting path - double feature channels at each downsampling
        self.contraction1 = downsampling(64, 128)
        self.contraction2 = downsampling(128, 256)
        self.contraction3 = downsampling(256, 512)
        self.contraction4 = downsampling(512, 1024)
        
        # Expanding path - half features channels at each upsampling
        self.expansion1 = upsampling(1024, 512, upsampling_mode=upsamplig_mode)
        self.expansion2 = upsampling(512, 256, upsampling_mode=upsamplig_mode)
        self.expansion3 = upsampling(256, 128, upsampling_mode=upsamplig_mode)
        self.expansion4 = upsampling(128, 64, upsampling_mode=upsamplig_mode)
        
        # Output process
        self.output = final(64, out_channels)
    
    def forward(self, x):
        x_init = self.input_process(x)
        x_down1 = self.contraction1(x_init)
        x_down2 = self.contraction2(x_down1)
        x_down3 = self.contraction3(x_down2)
        x_down4 = self.contraction4(x_down3)
        x = self.expansion1(x_down4, x_down3)
        x = self.expansion2(x, x_down2)
        x = self.expansion3(x, x_down1)
        x = self.expansion4(x, x_init)
        return self.output(x)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()