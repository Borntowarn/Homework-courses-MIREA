import wandb
import pandas as pd

from modules.dataset import HWTDataset
from modules.transforms import Transforms

from omegaconf import open_dict
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def modify_config(cfg: DictConfig) -> DictConfig:
    print('Initializing config...')
    
    if cfg.transforms.params.Grayscale.prob > 0:
        in_channels = cfg.transforms.params.Grayscale.params.num_output_channels
    else:
        in_channels = 3
    img_shape = cfg.transforms.params.Resize.params.size
    
    with open_dict(cfg):
        cfg.model.params.in_channels = in_channels
        cfg.model.params.img_shape = img_shape
            
    print('Config has successfully initialized!')
    return cfg


def init_wandb(cfg) -> None:
    if cfg.logging:
        print('Initializing WandB...')
        if cfg.id_resume:
            wandb.init(
                id=cfg.id_resume,
                project="Handwritten text recognition",
                resume='must'
            )
        else:
            wandb.init(
                project="Handwritten text recognition",
                name = f"{cfg.model.name}_{cfg.transforms.name}_{cfg.optim.name}_{cfg.scheduler.name}",
                config={
                    'Model': cfg.model.name,
                    'Transform': cfg.transforms.name,
                    'Optimizer': cfg.optim.name,
                    'Scheduler': cfg.scheduler.name if cfg.scheduler else 'None',
                    'architecture': 'RCNN',
                    'dataset': 'Handwritten Cyrillic dataset' if cfg.dataset == 'old_' else 'Custom dataset',
                    'epochs': cfg.epochs,
                }
            )
        print('WandB has successfully initialized!')


# pytorch-lightning DataModule
def load_data(cfg) -> tuple[HWTDataset, HWTDataset, set[str]]:
    transforms = Transforms(cfg.transforms.params)

    train_data = HWTDataset(cfg.train.dir,
                            cfg.train.labels,
                            transforms)
    test_data = HWTDataset(cfg.test.dir,
                           cfg.test.labels,
                           transforms)

    train_df = pd.read_csv(cfg.train.labels, delimiter='\t', names = ['Image name', 'Label'])
    alphabet = set(train_df['Label'].to_string()) - set('\n')
    
    train_dataloader = DataLoader(train_data, cfg.train_batch, True)
    test_dataloader = DataLoader(test_data, cfg.test_batch, True)

    return train_dataloader, test_dataloader, alphabet