import torch
import torch.nn as nn


class CGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, real, fake):
        loss = -1 * torch.mean(torch.log(real) + torch.log(1 - fake))
        if real > 1 or real < 0 or fake > 1 or fake < 0:
            print(f'D:{real = }\n  {fake = }')
            print(f'  {loss = }')
        return loss


class CGANGeneratorLoss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, fake):
        loss = -1 * torch.mean(torch.log(fake))  # Trick
        if fake < 0 or fake > 1:
            print(f'G:{fake = }')
            print(f'  {loss = }')
        return loss
