import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import random_split

torch.manual_seed(0)

class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size=batch_size
    
    def setup(self, stage=""):
        
        transform = transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST("./Data", train=True, download=True, transform=transform)
        self.train_set, self.val_set = random_split(dataset, [55000, 5000])
        
    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size)