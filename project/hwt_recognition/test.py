import torch
import hydra

from .modules.model import Model
from .modules.tokenizer import Tokenizer
from .modules.evaluator import Evaluator
from .utils import modify_config, load_data

from omegaconf import DictConfig
from pytorch_lightning import seed_everything

seed_everything(0, True)
CONFIG_PATH = './config'
CONFIG_NAME = 'config'


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig, path: str) -> None:
    cfg = modify_config(cfg) # Changing config
    train_dataloader, test_dataloader, alphabet = load_data(cfg) # Init loaders
    
    model = Model(**cfg.model.params)
    model.load_state_dict(torch.load(path))
    model = model.to(cfg.device)
    coder = Tokenizer(alphabet)

    evaluator = Evaluator(model, test_dataloader, coder, cfg.device)
    evaluator.evaluate(correcting=True)

if __name__ == '__main__':
    run('./file')

