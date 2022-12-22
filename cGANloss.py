import torch
import torch.nn as nn


class CGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, real, fake):
        # Possibility: compare final matrix to a block 1 for real, and a block 0 for fake instead of mean
        loss = -1 * torch.mean(torch.log(real) + torch.log(1 - fake))
        return loss


class CGANGeneratorLoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, fake):
        loss = -1 * torch.mean(torch.log(fake))  # Trick
        return loss
