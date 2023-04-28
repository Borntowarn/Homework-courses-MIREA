import os
import pytorch_lightning as pl

from typing import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataModule(pl.LightningDataModule):
    
    
    def __init__(
        self,
        root_dir: str,
        train_folder: str,
        test_folder: str,
        batch_size: int
    ) -> None:
        super(pl.LightningDataModule, self).__init__()
        
        self.train_dir = os.path.join(root_dir, train_folder)
        self.test_dir = os.path.join(root_dir, test_folder)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()])
        
        self.batch_size = batch_size
    
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_data = datasets.ImageFolder(self.train_dir, self.transform)
            self.len_train_dataloader = len(self.train_data) // self.batch_size
        if stage == 'test':
            self.test_data = datasets.ImageFolder(self.test_dir, self.transform)
    
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, self.batch_size, True, drop_last=True)

    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, self.batch_size, False, drop_last=True)