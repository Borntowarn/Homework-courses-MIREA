import torch

from .modules.model import Model
from .modules.tokenizer import Tokenizer
from .modules.evaluator import Evaluator
from .utils import load_config, load_data

from pytorch_lightning import seed_everything

seed_everything(0, True)
CONFIG_PATH = './config'
CONFIG_NAME = 'config'


def run(path) -> None:
    cfg = load_config() # Loading config
    train_dataloader, test_dataloader, alphabet = load_data(cfg) # Init loaders
    
    model = Model(**cfg.model.params)
    model.load_state_dict(torch.load(path))
    model = model.to(cfg.device)
    coder = Tokenizer(alphabet)

    evaluator = Evaluator(model, test_dataloader, coder, cfg.device)
    evaluator.evaluate(correcting=True)

if __name__ == '__main__':
    run('./file')

