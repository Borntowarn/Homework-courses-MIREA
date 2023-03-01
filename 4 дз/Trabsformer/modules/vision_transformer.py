import torch
from omegaconf import OmegaConf
import torch.nn as nn
import pytorch_lightning as pl

from typing import *
from hydra.utils import instantiate
from omegaconf import DictConfig
from .patch_embedding import PatchEmbedding
from .transformer import Transformer


class ViT(pl.LightningModule):
  
    def __init__(
        self,
        cfg: DictConfig,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        emb_len: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        drop_rate: int = 0.,
    ) -> None:
        super(ViT, self).__init__()
        self.save_hyperparameters(ignore=['loss_func', 'cfg'])
        
        # Path Embeddings, CLS Token, Position Encoding
        self.patch_embeddings = PatchEmbedding(img_size, patch_size, in_chans, emb_len)
        self.cls_token = nn.Parameter(torch.randn((1, 1, emb_len)))
        self.pos_encodings = nn.Parameter(torch.randn((self.patch_embeddings.num_patches + 1, emb_len)))

        # Transformer Encoder
        self.transformer = Transformer(num_layers, emb_len, num_heads, mlp_ratio, drop_rate)

        # Classifier
        self.classifier = nn.Linear(emb_len, num_classes)
        
        self.loss_func = instantiate(cfg.loss)
        self.cfg = cfg


    def forward(self, x):
        # Path Embeddings, CLS Token, Position Encoding
        b, c, h, w = x.shape
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = self.pos_encodings + torch.cat((cls_tokens, self.patch_embeddings(x)), dim = 1)

        # Transformer Encoder
        x = self.transformer(x)[:, 0, :].squeeze(1)

        # Classifier
        predictions = self.classifier(x)

        return predictions


    # Настраиваются параметры обучения
    def training_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        
        loss = self.loss_func(logits, targets)
        accuracy = torch.sum(logits.argmax(-1) == targets) / len(logits)
        
        lr = self.lr_schedulers().get_last_lr()[-1]
        self.log('loss', loss, on_epoch=True, on_step=False)
        self.log('acc', accuracy, on_epoch=True, on_step=False)
        self.log('Lr', lr, on_epoch=True, on_step=False)
        
        output = {
            'loss': loss,
            'acc': accuracy,
            'lr': lr
        }
        
        return output


    # Настраиваются параметры тестирования
    def test_step(self, batch, batch_idx):
        data, targets = batch
        logits = self(data)
        
        loss = self.loss_func(logits, targets)
        accuracy = torch.sum(logits.argmax(-1) == targets) / len(logits)
    
        self.log('Test acc', accuracy, prog_bar=True) 
        output = {
            'loss': loss,
            'acc': accuracy
        }
        return output


    # Конфигурируется оптимизатор
    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params = self.parameters())
        if self.cfg.scheduler.params.total_steps is not None:
            self.cfg.scheduler.params.total_steps = self.trainer.max_epochs * self.trainer.datamodule.len_train_dataloader
        scheduler = instantiate(self.cfg.scheduler.params, optimizer = optimizer)
        config = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': self.cfg.scheduler.step
            }
        }

        return config

    def training_epoch_end(self, outputs) -> None:
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        print(f'Эпоха {self.current_epoch}, loss = {loss}')