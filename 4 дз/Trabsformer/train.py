import hydra
import torch

from typing import *
from dataclasses import dataclass
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore


@dataclass
class DataModule:
    _target_: str = 'modules.datamodule.DataModule'
    root_dir: str = 'dataset/'
    train_folder: str = 'validation'
    test_folder: str = 'test/'
    batch_size: int = 64
    len_train_dataset: int = 3000


@dataclass
class WandbLogger:
    _target_: str = 'pytorch_lightning.loggers.WandbLogger'
    project: str = 'First ViT'
    log_model: bool = True
    name: str = 'Run name'


@dataclass
class Trainer:
    _target_: str = 'pytorch_lightning.trainer.Trainer'
    accelerator: str = 'gpu'
    max_epochs: int = 10
    default_root_dir: str = './lightning'
    logger: Optional[str] = None


@dataclass
class VisionTransformer:
    _target_: str = 'modules.vision_transformer.ViT'
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    emb_len: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    drop_rate: float = 0.0


@dataclass
class General:
    epochs: int = 100
    logging: bool = False
    device: str = 'gpu'


@dataclass
class Config:
    name: str = 'Default config'
    
    model: VisionTransformer = VisionTransformer()
    trainer: Trainer = Trainer()
    datamodule: DataModule = DataModule()
    general: General = General()


cs = ConfigStore.instance()
cs.store(name='config', node=Config)
cs.store(group='model', name="Model1", node=VisionTransformer)
cs.store(group='trainer', name="Trainer", node=Trainer)
cs.store(group='loggers', name='WandbLogger', node=WandbLogger)
cs.store(group='datamodule', name='DataModule', node=DataModule)


@hydra.main(config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    
    model = instantiate(cfg.model, cfg=cfg)
    datamodule = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    
    trainer.fit(model, datamodule)

run()