import os
import torch
import pandas as pd

from PIL import Image
from typing import Optional
from torchvision import transforms
from torch.utils.data import Dataset

class HWTDataset(Dataset):
    """
    Class for creating custom image2label dataset from folder

    Args:
        root_dir (str): Path to image dir
        label_dir (str): Path to labling file
        transforms (Optional[transforms.Compose], optional): Transforms you want to apply. Defaults to None.
    """
    
    
    def __init__(
        self,
        root_dir: str,
        label_dir: str,
        transforms: Optional[transforms.Compose] = None
    ) -> None:
        super(HWTDataset, self).__init__()

        # Loading labling file
        name_label = pd.read_csv(label_dir, delimiter='\t', names = ['Image name', 'Label'])
        name_label['Image name'] = name_label['Image name'].apply(lambda x: os.path.join(root_dir, x))
        self.data = name_label.to_dict('split')['data']
        
        self.transforms = transforms
    
    
    def __len__(self) -> int:
        return len(self.data)
    
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        
        path, label = self.data[index]
        img = Image.open(path).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, label