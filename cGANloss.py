import torch
import torch.nn as nn


class CGANGeneratorLoss(nn.Module):
    def __init__(self):
        super(CGANGeneratorLoss, self).__init__()

    def __call__(self, real, fake):
        loss = torch.mean(torch.log(real) + torch.log(1 - fake))
        return loss


class CGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super(CGANDiscriminatorLoss, self).__init__()

    def __call__(self, fake):
        loss = -1 * torch.mean(torch.log(fake))  # Trick
        return loss
