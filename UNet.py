import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F
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
    
    def __init__(self, in_channels, out_channels, upsampling_mode, output_input):
        super().__init__()
        self.upsamplig_mode = upsampling_mode
        self.concat_mode = output_input
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsampling_mode), #mode can be:'nearest','linear','bilinear','bicubic','trilinear'
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        ) #Another possibility for upsampling is using an inverse conv with nn.ConvTranspose2d
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
        
        if self.concat_mode == 'different':
            # Crop contracting path outputs and concat with expanding map
            H = x.size(dim=2)
            W = x.size(dim=3)  
            top = (x_down.size(dim=2) - H)//2
            left = (x_down.size(dim=3) - W)//2

            cropped_x_down = crop(x_down, top, left, H, W)
            x = cat((x, cropped_x_down), dim=1) #check dim
            
        elif self.concat_mode == 'same':
            # Pad expanding path outputs and concat with contracting map
            H_diff = x_down.size(dim=2) - x.size(dim=2)
            H_left = H_diff//2
            W_diff = x_down.size(dim=3) - x.size(dim=3)
            W_left = W_diff//2
            padded_x = F.pad(x, (W_left, (W_diff - W_left), H_left, (H_diff - H_left)))
            x = cat((padded_x, x_down), dim=1)
        else:
            raise Exception(f"{self.concat_mode} is not a valid input for output_input")
        
        return self.double_conv(x)
    
# Final layer (class mapping)
class final(nn.Module):
    
    def __init__(self, in_channels, n_outputs, output_size, upsampling_mode, output_input):
        super().__init__()
        self.output_input = output_input
        
        self.output_same = nn.Sequential(
            nn.Upsample(size=output_size, mode=upsampling_mode),
            nn.Conv2d(in_channels, n_outputs, kernel_size=1)
        )
        
        self.output_valid = nn.Conv2d(in_channels, n_outputs, kernel_size=1)
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
        # For output_size = input_size
        if self.output_input == 'same':
            return self.output_same
        # If it is okay for output and input sizes being different
        elif self.output_input == 'different':
            return self.output(x) 
        else:
            raise Exception(f"{self.output_input} is not a valid input for output_input")

# UNet
class UNet(nn.Module):
    def __init__(self, output_size, in_channels=1, out_channels=3, upsampling_mode='nearest', output_input='same'):
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
        self.expansion1 = upsampling(1024, 512, upsampling_mode=upsampling_mode, concat_mode=concat_mode)
        self.expansion2 = upsampling(512, 256, upsampling_mode=upsampling_mode, concat_mode=concat_mode)
        self.expansion3 = upsampling(256, 128, upsampling_mode=upsampling_mode, concat_mode=concat_mode)
        self.expansion4 = upsampling(128, 64, upsampling_mode=upsampling_mode, concat_mode=concat_mode)
        
        # Output process
        self.output = final(64, out_channels, upsampling_mode=upsampling_mode, output_size=output_size)
        
        # Dropout
        self.init_dropout = nn.Dropout2d(p=0.2)
        self.final_dropout = nn.Dropout2d(p=0.2)
    
    def forward(self, x):
        x_init = self.input_process(x)
        # Dropout as noise
        x_init = self.init_dropout(x_init)
        x_down1 = self.contraction1(x_init)
        x_down2 = self.contraction2(x_down1)
        x_down3 = self.contraction3(x_down2)
        x_down4 = self.contraction4(x_down3)
        x = self.expansion1(x_down4, x_down3)
        x = self.expansion2(x, x_down2)
        x = self.expansion3(x, x_down1)
        x = self.expansion4(x, x_init)
        # Dropout as noise
        #x = self.final_dropout(x)
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