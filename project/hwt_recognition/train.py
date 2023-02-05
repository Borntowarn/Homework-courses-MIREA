import torch
import hydra

from hydra.utils import instantiate
from modules.model import Model
from modules.trainer import Trainer
from modules.tokenizer import Tokenizer
from utils import modify_config, init_wandb, load_data

from omegaconf import DictConfig
from pytorch_lightning import seed_everything


seed_everything(0, True)
CONFIG_PATH = './config'
CONFIG_NAME = 'config'

@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    cfg = modify_config(cfg) # Changing config
    init_wandb(cfg) # Init WandB
    train_dataloader, test_dataloader, alphabet = load_data(cfg) # Init loaders
    
    # Init every class for training
    model = Model(**cfg.model.params)
    model = model.to(cfg.device)
    tokenizer = Tokenizer(alphabet)
    ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
    optimizer = instantiate(cfg.optim.params, params=model.parameters())
    
    if cfg.scheduler:
        cfg.scheduler.params.total_steps *= len(train_dataloader)
        scheduler = instantiate(cfg.scheduler.params, optimizer=optimizer)
    else: 
        scheduler = None
    
    trainer = Trainer(model, optimizer, train_dataloader, ctc_loss, tokenizer, cfg.epochs,
                      f'{cfg.model.name}_{cfg.transforms.name}_{cfg.optim.name}_{cfg.scheduler.name}',
                      alphabet, scheduler, cfg.logging, cfg.device)
    
    trainer.train()

if __name__ == '__main__':
    run()