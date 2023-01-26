import torch
import torch.nn as nn
import torch.nn.functional as f


class PatchGANConvBlock(nn.Module):
    def __init__(self, filters, kernel_size=4, stride=4, padding=1):
        super(PatchGANConvBlock, self).__init__()
        self.conv = nn.Conv2d(filters[0], filters[1], kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(num_features=filters[1])

    def _init_weights(self, module):        
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = nn.LeakyReLU(0.2, True)(out)
        #print(f'ConvBlock {out.size() = }')
        return out


class PatchGAN(nn.Module):
    def __init__(self, color_channels=3, patch_size=64, kernel_size=4, stride=2, padding=1):
        super(PatchGAN, self).__init__()
        # Find layers to have a receptive field of 64 x 64 for each output element, given kernel_size, stride, padding
        # R(k) = R(k-1) + 3 * 4 ** (k - 1), Receptive Field for layer k, given kernel size 4 and stride 4.
        # R(k) = R(k - 1) + (kernel_size(k) - 1) * ∏_i^(k-1) stride(i), more generally
        nf = [color_channels + 1, 64, 128, 256, 512, 1]
        n_layers = len(nf) - 2
        layers = []
        for n in range(n_layers - 1):
            layers += [PatchGANConvBlock((nf[n], nf[n + 1]), kernel_size=kernel_size, stride=stride, padding=padding)]
        layers += [PatchGANConvBlock((nf[-3], nf[-2]), kernel_size=kernel_size, stride=1, padding=padding)]
        layers += [nn.Conv2d(nf[-2], nf[-1], kernel_size=kernel_size, stride=1, padding=padding)]
        self.model = nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module)
            if module.bias is not None:
                module.bias.data.zero_()   

    def forward(self, colored, grayscale):
        #print(f'Input {colored.size() = }')
        out = self.model(torch.cat((colored, grayscale), dim=1))
        out = torch.sigmoid(out)
        #print(f'PatchGAN {out.size() = }')
        return out
