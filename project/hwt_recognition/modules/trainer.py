import wandb
import torch

from .tokenizer import Tokenizer

from torch import nn
from tabulate import tabulate
from tqdm.notebook import tqdm
from typing import Iterable, Optional, List, Tuple
from torchmetrics import CharErrorRate, WordErrorRate

class Trainer:
    """
    Class for training models

    Args:
        model (nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        dataloader (torch.utils.data.Dataloader): _description_
        lossfunc (nn.Module): _description_
        tokenizer (Tokenizer): _description_
        epochs (int): _description_
        model_name (str): _description_
        train_alphabet (Iterable[str]): _description_
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): _description_. Defaults to None.
        logging (Optional[bool], optional): _description_. Defaults to False.
        device (Optional[str], optional): _description_. Defaults to 'cuda'.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        lossfunc: nn.Module,
        tokenizer: Tokenizer,
        epochs: int,
        model_name: str,
        train_alphabet: Iterable[str],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logging: Optional[bool] = False,
        device: Optional[str] = 'cuda'
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.lossfunc = lossfunc
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.model_name = model_name
        self.logging = logging
        self.device = torch.device(device)
        self.train_alphabet = train_alphabet


    def print_epoch_data(
        self,
        epoch: int,
        mean_loss: float,
        char_error: float,
        word_error: float,
    ) -> None:
        """
        This method prints epoch stat

        Args:
            epoch (int): Epoch number
            mean_loss (float): Mean loss
            char_error (float): CER
            word_error (float): WER
        """
        print(tabulate(
            [['epoch', 'mean loss', 'mean cer', 'mean wer'],
             [epoch, round(mean_loss, 4), round(char_error, 4), round(word_error, 4)]],
            headers='firstrow',
            tablefmt='fancy_grid'))


    def save_model(self, mean_loss: float, char_error: float) -> None:
        """
        This method saves model
        Args:
            mean_loss (float): Mean loss
            char_error (float): CER
        """
        torch.save(self.model.state_dict(),
                    f'./{self.model_name} \
                    _L-{round(mean_loss, 4)} \
                    _CER-{round(char_error, 4)}.pth')


    def log(self, mean_loss: float, char_error: float, word_error: float) -> None:
        """
        This method logs stat in WandB

        Args:
            mean_loss (float): Mean loss
            char_error (float): CER
            word_error (float): WER
        """
        wandb.log({'loss': mean_loss,
                    'CER': char_error,
                    'WER': word_error,
                    'Learn Rate': 
                        self.scheduler.get_last_lr()[-1] 
                        if self.scheduler 
                        else self.optimizer.param_groups[0]['lr']})
    
    
    def print_save_stat(
        self, 
        outputs: List[List[float]],
        epoch: int
    ) -> None:
        """
        This method helps to log and save model

        Args:
            outputs (List[List[float]]): 
                The first list is list with losses, the second is CER, the third is WER by epoch
            epoch (int): Number of epoch
        """
        output = torch.Tensor(outputs)
        mean_loss = output[:, 0].mean().item()
        char_error = output[:, 1].mean().item()
        word_error = output[:, 2].mean().item()
        
        self.print_epoch_data(epoch, mean_loss, char_error, word_error)
        
        if self.logging:
            self.log(mean_loss, char_error, word_error)
        
        if mean_loss < 0.1 \
           or not (epoch + 1) % 5 \
           or epoch == self.epochs:
            self.save_model(mean_loss, char_error)
    
    def forward(
        self, data: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model

        Args:
            data (torch.Tensor): Input of model
            targets (torch.Tensor): True labels

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Loss val, log_softmax probs, sequence length for every img in batch
        """
        self.optimizer.zero_grad()
        classes, lengths = self.tokenizer.encode(targets)
        data = data.to(self.device)
        classes = classes.to(self.device)
        
        logits = self.model(data)
        logits = logits.contiguous().cpu()
        L, N, H_out = logits.size() # L - seq length, N - batch size, H_out - len alphabet
        pred_sizes = torch.LongTensor([L for i in range(N)]).to(self.device)
        classes = classes.view(-1).contiguous()
        loss = self.lossfunc(logits, classes, pred_sizes, lengths)
        
        return loss, logits, pred_sizes
    
    
    def decode(self, logits: torch.Tensor, pred_sizes: torch.Tensor) -> List[str]:
        """
        This method decode predicted probs to phrases

        Args:
            logits (torch.Tensor): Model output
            pred_sizes (torch.Tensor): Sequence length for every img in batch

        Returns:
            list[str]: Decoded phrases
        """
        probs, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.tokenizer.decode(preds.data, pred_sizes.data)
        return sim_preds


    def backward(self, loss: torch.Tensor) -> None:
        """
        Backward pass of the model

        Args:
            loss (torch.Tensor): Loss value
        """
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
    
    def statistics(self, loss: torch.Tensor, sim_preds: List[str], targets: List[str]) -> List[float]:
        """
        Calculate stat of batch

        Args:
            loss (torch.Tensor): Loss value
            sim_preds (List[str]): Predicted phrases
            targets (List[str]): True phrases

        Returns:
            list[float]: loss, CER, WER
        """
        CER = CharErrorRate()
        WER = WordErrorRate()
        cer = CER(sim_preds, targets)
        wer = WER(sim_preds, targets)
        
        return [abs(loss.item()), cer, wer]
    
    
    def train(self) -> nn.Module:
        """
        This method run training

        Returns:
            torch.Module: Trained model
        """
        self.model.train()
        
        if self.logging:
            wandb.watch(self.model, self.lossfunc, log='all', log_freq=100)
        
        for epoch in tqdm(range(1, self.epochs + 1), total=self.epochs):
            outputs = []
            for (data, targets) in tqdm(self.dataloader, total=len(self.dataloader)):
                
                loss, logits, pred_sizes = self.forward(data, targets)
                sim_preds = self.decode(logits, pred_sizes)
                self.backward(loss)
                outputs.append(self.statistics(loss, sim_preds, targets))
            
            self.print_save_stat(outputs, epoch)
        
        return self.model