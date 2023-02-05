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
PATH_TO_MODEL = ''


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    cfg = modify_config(cfg) # Changing config
    train_dataloader, test_dataloader, alphabet = load_data(cfg) # Init loaders
    
    model = Model(**cfg.model.params)
    model.load_state_dict(torch.load(PATH_TO_MODEL))
    model = model.to(cfg.device)
    tokenizer = Tokenizer(alphabet)

    evaluator = Evaluator(model, test_dataloader, tokenizer, cfg.device)
    evaluator.evaluate()

if __name__ == '__main__':
    run()

