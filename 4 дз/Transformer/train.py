import hydra

import os
from typing import *
from dataclasses import dataclass, field
from omegaconf import DictConfig
from hydra.utils import instantiate
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

@dataclass
class General:
    epochs: int = 100
    batch_size: int = 4
    max_lr: float = 0.0004
    logging: bool = False
    device: str = 'gpu'
    default_dir: str = '.'

@dataclass
class Config:
    name: str = 'Any run'
    
    model: Any = MISSING
    scheduler: Any = MISSING
    optimizer: Any = MISSING
    loss: Any = MISSING
    trainer: Any = MISSING
    datamodule: Any = MISSING
    logger: Any = MISSING
    general: General = field(default_factory=General)


cs = ConfigStore().instance()
cs.store(name='config', node=Config)
cs.store(name='general', node=General)

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:

    model = instantiate(cfg.model, cfg)
    datamodule = instantiate(cfg.datamodule)
    logger = False
    if cfg.general.logging:
        logger = instantiate(cfg.logger)
        logger.watch(model, log = 'all', log_freq=100)
    trainer = instantiate(cfg.trainer, logger=logger)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == '__main__':
    run()