import torch

from .modules.Model import Model
from .modules.Trainer import Trainer
from .modules.Decoder import SymbolCoder
from .utils import load_config, init_wandb, load_data

from pytorch_lightning import seed_everything


seed_everything(0, True)
CONFIG_PATH = './config'
CONFIG_NAME = 'config'


def run() -> None:
    cfg = load_config() # Loading config
    init_wandb(cfg) # Init WandB
    train_dataloader, test_dataloader, alphabet = load_data(cfg) # Init loaders
    
    # Init every class for training
    model = Model(**cfg.model.params)
    model = model.to(cfg.device)
    coder = SymbolCoder(alphabet)
    ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
    optimizer = getattr(torch.optim, cfg.optim.optim)(model.parameters(), **cfg.optim.params)
    
    if cfg.scheduler:
        cfg.scheduler.params.total_steps *= len(train_dataloader)
        scheduler = getattr(torch.optim.lr_scheduler,
                            cfg.scheduler.scheduler)(optimizer,
                                                     **cfg.scheduler.params)
    else: scheduler = None
    
    trainer = Trainer(model, optimizer, train_dataloader, ctc_loss, coder, cfg.epochs,
                      f'{cfg.model.name}_{cfg.transforms.name}_{cfg.optim.name}_{cfg.scheduler.name}',
                      alphabet, scheduler, cfg.logging, cfg.device)
    
    trainer.train()

if __name__ == '__main__':
    run()