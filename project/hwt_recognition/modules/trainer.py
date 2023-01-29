import wandb
import torch

from model import Model

from torch import Tensor
from tabulate import tabulate
from tqdm.notebook import tqdm
from torchmetrics import CharErrorRate, WordErrorRate

class Trainer:
    
    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        lossfunc,
        coder,
        epochs,
        model_name,
        train_alphabet,
        scheduler = None,
        logging : bool = False,
        device : str = 'cuda'
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.lossfunc = lossfunc
        self.coder = coder
        self.epochs = epochs
        self.model_name = model_name
        self.LOGGING = logging
        self.DEVICE = device
        self.train_alphabet = train_alphabet


    def print_epoch_data(
        self,
        epoch: int,
        mean_loss: float,
        char_error: float,
        word_error: float,
        zero_out_losses: float
    ) -> None:
        print(tabulate(
            [['epoch', 'mean loss', 'mean cer', 'mean wer', 'zero loss warnings'],
             [epoch, round(mean_loss, 4), round(char_error, 4),
              round(word_error, 4), zero_out_losses]],
            headers='firstrow',
            tablefmt='fancy_grid'))


    def save_model(self, mean_loss: float, char_error: float) -> None:
        torch.save(self.model.state_dict(),
                    f'./{self.model_name} \
                    _L-{round(mean_loss, 4)} \
                    _CER-{round(char_error, 4)}.pth')


    def log(self, mean_loss: float, char_error: float, word_error: float) -> None:
        wandb.log({'loss': mean_loss,
                    'CER': char_error,
                    'WER': word_error,
                    'Learn Rate': 
                        self.scheduler.get_last_lr()[-1] 
                        if self.scheduler 
                        else self.optimizer.param_groups[0]['lr']})
    
    
    def print_save_stat(self, outputs: list, epoch: int, zero_out: int) -> None:
        
        assert len(outputs) != 0, 'Error: bad loss'
            
        output = torch.Tensor(outputs)
        mean_loss = output[:, 0].mean().item()
        char_error = output[:, 1].mean().item()
        word_error = output[:, 2].mean().item()
        
        self.print_epoch_data(epoch, mean_loss, char_error, word_error, zero_out)
        
        if self.LOGGING:
            self.log(mean_loss, char_error, word_error)
        
        if mean_loss < 0.1 or not (epoch + 1) % 5:
            self.save_model(mean_loss, char_error)
    
    def forward(self, data: torch.Tensor, targets: Tensor) -> None:
        self.optimizer.zero_grad()
        classes, lengths = self.coder.encode(targets)
        data = data.to(self.DEVICE)
        classes = classes.to(self.DEVICE)
        
        logits = self.model(data)
        logits = logits.contiguous().cpu()
        T, N, C = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(N)]).to(self.DEVICE)
        classes = classes.view(-1).contiguous()
        loss = self.lossfunc(logits, classes, pred_sizes, lengths)
    
    
    def prediction(self, logits: Tensor, pred_sizes: Tensor) -> list[str]:
        probs, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.coder.decode(preds.data, pred_sizes.data)
        return sim_preds


    def backward(self, loss: Tensor, sim_preds: list, targets: list) -> tuple[float, float]:
        CER = CharErrorRate()
        WER = WordErrorRate()
        cer = CER(sim_preds, targets)
        wer = WER(sim_preds, targets)
        
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        return cer, wer
    
    
    def statistics(self, loss: Tensor, cer: float, wer: float) -> list[float, float, float]:
        return [abs(loss.item()), cer, wer]
    
    
    def train(self) -> Model:
        self.model.train()
        
        if self.LOGGING:
            wandb.watch(self.model, self.lossfunc, log='all', log_freq=100)
        
        for epoch in tqdm(range(self.epochs), total=self.epochs):
            zero_out_losses = 0
            outputs = []
            for (data, targets) in tqdm(self.dataloader, total=len(self.dataloader)):
                
                loss, logits, pred_sizes = self.forward(data, targets)
                sim_preds = self.prediction(logits, pred_sizes)
                cer, wer = self.backward(loss, sim_preds, targets)
                outputs.append(self.statistics(loss, cer, wer))
            
            self.print_save_stat(outputs, epoch, zero_out_losses)
        
        return self.model