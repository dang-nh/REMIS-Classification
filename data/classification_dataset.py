import os.path
from base import BaseDataLoader
from torchvision.datasets import DatasetFolder
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class ClassificationDataset(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.data_dir = data_dir
        self.dataset = DatasetFloder('datasets', transforms=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)