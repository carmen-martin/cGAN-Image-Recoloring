import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import crop

# Downsampling
class downsampling(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation=True, normalization=True):
        super().__init__()
        
        self.activation = activation
        self.normalization = normalization
        
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        if activation:
            self.act = nn.LeakyReLU(0.2)
        if normalization:
            self.norm = nn.BatchNorm2d(out_channels)
        
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
        x = self.downsample(x)
        
        if self.normalization:
            x = self.norm(x)
        if self.activation:
            x = self.act(x)
        
        return x


# Upsampling (doubleconv+upsampling)
class upsampling(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation=True, normalization=True, dropout=False):
        super().__init__()
        self.activation = activation
        self.normalization = normalization
        self.dropout = dropout
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        if activation:
            self.act = nn.ReLU()
        if normalization:
            self.norm = nn.BatchNorm2d(out_channels)
        if dropout:
            self.drop = nn.Dropout2d(p=0.5)
        
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
        if self.activation:
            x = self.act(x)
            
        x = self.upsample(x)
        
        if self.normalization:
            x = self.norm(x)
        if self.dropout:
            x = self.drop(x)
        
        H_x = x.size(dim=2)
        W_x = x.size(dim=3)
        H_xd = x_down.size(dim=2)
        W_xd = x_down.size(dim=3)
        
        if H_x == H_xd and W_x == W_xd:
            x = cat((x, x_down), dim=1) #check if this is possible with dims or we need to crop/pad
        else:
            # Pad expanding path outputs and concat with contracting map
            H_diff = H_xd - H_x
            H_left = H_diff//2
            W_diff = W_xd - W_x
            W_left = W_diff//2
            padded_x = F.pad(x, (W_left, (W_diff - W_left), H_left, (H_diff - H_left)))
            x = cat((padded_x, x_down), dim=1)
        
        return x
    
# Final layer (class mapping)
class final(nn.Module):
    
    def __init__(self, in_channels, n_outputs):
        super().__init__()
        
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, n_outputs, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
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
        return self.output(x)

# UNet
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Input process
        self.input_process = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        
        # Contracting path - double feature channels at each downsampling
        self.contraction1 = downsampling(64, 128, normalization=False)
        self.contraction2 = downsampling(128, 256)
        self.contraction3 = downsampling(256, 512)
        self.contraction4 = downsampling(512, 512)
        self.contraction5 = downsampling(512, 512)
        self.contraction6 = downsampling(512, 512)
        self.contraction7 = downsampling(512, 512)
        
        self.mid_process = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        
        # Expanding path - half features channels at each upsampling
        self.expansion1 = upsampling(512, 512, dropout=True)
        self.expansion2 = upsampling(1024, 512, dropout=True)
        self.expansion3 = upsampling(1024, 512, dropout=True)
        self.expansion4 = upsampling(1024, 512)
        self.expansion5 = upsampling(1024, 256)
        self.expansion6 = upsampling(512, 128)
        self.expansion7 = upsampling(256, 64)
        
        # Output process
        self.output = final(128, out_channels)
    
    def forward(self, x):
        x_init = self.input_process(x)
        print(f'{x_init.shape = }')
        x_down1 = self.contraction1(x_init)
        print(f'{x_down1.shape = }')
        x_down2 = self.contraction2(x_down1)
        print(f'{x_down2.shape = }')
        x_down3 = self.contraction3(x_down2)
        print(f'{x_down3.shape = }')
        x_down4 = self.contraction4(x_down3)
        print(f'{x_down4.shape = }')
        x_down5 = self.contraction5(x_down4)
        print(f'{x_down5.shape = }')
        x_down6 = self.contraction6(x_down5)
        print(f'{x_down6.shape = }')
        x_down7 = self.contraction7(x_down6)
        print(f'{x_down7.shape = }')
        x_mid = self.mid_process(x_down7)
        print(f'{x_mid.shape = }')
        x = self.expansion1(x_mid, x_down6)
        print(f'{x.shape = }')
        x = self.expansion2(x, x_down5)
        print(f'{x.shape = }')
        x = self.expansion3(x, x_down4)
        print(f'{x.shape = }')
        x = self.expansion4(x, x_down3)
        print(f'{x.shape = }')
        x = self.expansion5(x, x_down2)
        print(f'{x.shape = }')
        x = self.expansion6(x, x_down1)
        print(f'{x.shape = }')
        x = self.expansion7(x, x_init)
        print(f'{x.shape = }')
        print(f'{self.output(x).shape = }')
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