import torch
import os
import torchvision
from torch.utils.data import Dataset, Subset, DataLoader
from pycocotools.coco import COCO

DATAPATH = '/mnt/NeuralNetworksDL/coco/'
GRAYS = [498856, 6432, 84582, 457741, 11801, 427401, 821, 225717, 118895, 325387, 217886, 575029,
578250, 81003, 100896, 150354, 476888, 436984, 122051, 155083, 156878, 61048, 105872,
233263, 406404, 416869, 518025, 343009, 416372, 140627, 207339, 5294, 300200, 72098, 492325,
507794, 211867, 577207, 249711, 173610, 563447, 257178, 525513, 221691, 154053, 470442, 296884,
104124, 32405, 384907, 394322, 176397, 85407, 491058, 389984, 560349, 434837, 220770, 451074, 86,
406011, 406744, 134071, 269858, 410498, 53756, 46433, 363331, 280731, 140623, 204792, 80906, 33127,
132791, 228474, 571415, 361221, 208206, 342051, 349069, 377984, 155954, 451095, 532787, 573179,
155811, 27412, 124694, 336668, 577265, 185639, 103499, 532919, 510587, 145288, 559665, 176483, 342921,
64270, 123539, 205782, 205486, 57978, 353952, 312288, 397575, 439589, 431115, 126531, 287422,
555583, 173081, 380088, 401901, 579138, 260962, 166522, 426558, 421195, 361516, 390663, 15236, 30349, 107450, 385625, 29275, 443909, 250239, 134206, 226585, 518951, 131942, 1350, 93120, 509358, 561842, 131366,
386204, 268036, 217341, 6379, 549879, 564314, 111109, 434765, 35880, 381270, 330736, 384693, 39068, 18702,
316867, 186888, 264165, 389206, 15286, 445845, 58517, 470933, 33352, 210847, 458073, 377837, 293833,
25404, 95753, 270925, 463454, 443689, 213280, 563376, 77709, 243205, 313608, 210175, 566596, 60060,
259284, 263002, 576700, 484742, 66642, 341892, 400107, 394547, 12345, 75052, 39790, 369966, 134918,
505962, 39900, 179405, 34861, 220898, 450674, 223616, 454000, 540378, 3293, 492395, 249835, 429633,
520479, 579239, 537427, 449901, 358281, 384910, 494273, 140092, 321897, 347111, 571503, 503640, 64332,
421613, 113929, 10125, 8794, 107962, 496444, 480482, 264753, 87509, 40428, 517899]


class CocoDataSet(Dataset):
    def __init__(self, annt_file, data_dir=DATAPATH, cats=None, size=5000, transform=None):
        self.data_dir = data_dir
        coco = COCO(data_dir + 'annotations/' + annt_file)
        imgIds = coco.getImgIds(catIds=coco.getCatIds(catNms=cats))
        self.filenames = ['{:012}'.format(x) + '.jpg' for x in imgIds if x not in GRAYS]
        self.transform = transform
        if size > len(self.filenames):
            self.size = len(self.filenames)
        else:
            self.size = size
        print('Dataset size:', self.size)

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


def load_coco_dataset(batch_size=64, cats=None, size=123176, dim=64):
    # ImageNet normalization, Resizing to dim x dim
    data = CocoDataSet('instances_train2017.json',
                       cats=cats,
                       data_dir=DATAPATH,
                       size=size,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                           torchvision.transforms.Resize((dim, dim))]))
    n_val = int(0.1 * len(data)) + 1
    idx = torch.randperm(len(data))
    train_dataset = Subset(data, idx[:-n_val])
    valid_dataset = Subset(data, idx[-n_val:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader
