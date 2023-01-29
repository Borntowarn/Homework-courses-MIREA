from omegaconf import OmegaConf
from torchvision import transforms

class Transforms(transforms.Compose):
    
    def __init__(self, args) -> None:
        
        self.transforms = []
        
        for key, value in args.items():
            value = OmegaConf.to_object(value)
            self.transforms.append(
                transforms.RandomApply([
                    getattr(transforms, key)(**value['params'])], # Transform
                    value['prob']) # Probability of apply
                )
        self.transforms.append(transforms.ToTensor())