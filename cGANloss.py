import torch
import torch.nn as nn


class CGANDiscriminatorLoss(nn.Module):
    def __init__(self, device, smoothing=0.05):
        self.device = device
        self.smoothing = smoothing

    def __call__(self, real, fake):
        loss = 0.5 * (nn.BCELoss()(fake, torch.zeros(fake.size()).to(device=self.device)) +\
                      nn.BCELoss()(real, torch.ones(real.size()).to(device=self.device) - self.smoothing)) # One-sided smoothing
        return loss


class CGANGeneratorLoss(nn.Module):
    def __init__(self, device, reg):
        self.device = device
        self.reg = reg

    def __call__(self, fake, colored, g_output):
        loss = -1 * nn.BCELoss()(fake, torch.ones(fake.size()).to(device=self.device)) +\
               self.reg * nn.L1Loss()(colored, g_output) # Training Trick
        return loss
