import torch
import os
import torchvision
from torch.utils.data import Dataset, Subset, DataLoader

DATAPATH = '/mnt/NeuralNetworksDL/coco/'


class CocoDataSet(Dataset):
    def __init__(self, data_dir, size=5000, transform=None):
        self.data_dir = data_dir
        self.size = size
        self.filenames = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # __getitem__ actually reads the img content
        colored = torchvision.io.read_image(self.data_dir + self.filenames[index]).to(torch.float32) / 255
        if self.transform:
            colored = self.transform(colored)
        grayscale = torchvision.transforms.functional.rgb_to_grayscale(colored)
        return colored, grayscale


def load_coco_dataset(batch_size):
    # ImageNet normalization, Resizing to 256 x 256
    data = CocoDataSet(DATAPATH,
                       size=123176,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                           torchvision.transforms.Resize((256, 256))]))
    n_val = int(0.2 * len(data)) + 1
    idx = torch.randperm(len(data))
    train_dataset = Subset(data, idx[:-n_val])
    valid_dataset = Subset(data, idx[-n_val:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader
