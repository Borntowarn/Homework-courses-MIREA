from .modules.model import Model
from .modules.tokenizer import Tokenizer
from .modules.transforms import Transforms
from .modules.recognizer import Recognizer
from .utils import load_config, init_wandb, load_data

from pytorch_lightning import seed_everything


seed_everything(0, True)
CONFIG_PATH = './config'
CONFIG_NAME = 'config'


def run() -> None:
    cfg = load_config() # Loading config
    init_wandb(cfg) # Init WandB
    train_dataloader, test_dataloader, alphabet = load_data(cfg) # Init loaders
    
    model = Model(**cfg.model.params)
    model = model.to(cfg.device)
    coder = Tokenizer(alphabet)
    
    recognizer = Recognizer(model, coder, Transforms(cfg.transforms.params), cfg.device)
    recognizer.recognize_from_painted()
    
if __name__ == '__main__':
    run()