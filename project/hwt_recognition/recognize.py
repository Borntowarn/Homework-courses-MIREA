import hydra

from modules.model import Model
from modules.tokenizer import Tokenizer
from modules.transforms import Transforms
from modules.recognizer import Recognizer
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
    
    model = Model(**cfg.model.params)
    model = model.to(cfg.device)
    coder = Tokenizer(alphabet)
    
    recognizer = Recognizer(model, coder, Transforms(cfg.transforms.params), cfg.device)
    recognizer.recognize_from_painted()
    
if __name__ == '__main__':
    run()