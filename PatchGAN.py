import torch
import torch.nn as nn
import torch.nn.functional as f


class PatchGANConvBlock(nn.Module):
    def __init__(self, filters, kernel_size=4, stride=4, padding=1):
        super(PatchGANConvBlock, self).__init__()
        self.conv = nn.Conv2d(filters[0], filters[1], kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(num_features=filters[1])

    def forward(self, input):
        out = self.norm(self.conv(input))
        out = f.ReLU(out)
        return out


class PatchGAN(nn.Module):
    def __init__(self, patch_size=64, kernel_size=4, stride=4, padding=1):
        super(PatchGAN, self).__init__()
        # Find layers to have a receptive field of 64 x 64 for each output element, given kernel_size, stride, padding
        # R(k) = R(k-1) + 3 * 4 ** (k - 1), Receptive Field for layer k, given kernel size 4 and stride 4.
        # R(k) = R(k - 1) + (kernel_size(k) - 1) * ∏_i^(k-1) stride(i), more generally
        n_layers = 1
        while 4 ** n_layers > patch_size:
            n_layers += 1
        nf = 1
        layers = []
        for n in range(n_layers):
            nf *= n + 1
            layers += [PatchGANConvBlock((nf, min(nf * n, 8)), kernel_size=kernel_size, stride=stride, padding=padding)]
        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=stride, padding=padding)]
        self.model = nn.Sequential(*layers)

    def forward(self, image):
        out = torch.mean(self.model(image))
        return out
