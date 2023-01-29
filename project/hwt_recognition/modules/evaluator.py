import torch

from .corrector import Corrector
from .tokenizer import Tokenizer

from tqdm.notebook import tqdm
from typing import Dict, List, Tuple, Optional
from torchmetrics import CharErrorRate, WordErrorRate

class Evaluator:
    """
    Class for evaluate CER, WER of model and
    count stat about symbols errors.

    Args:
        model (torch.Module): Model for evaluating
        loader (torch.utils.data.Dataloader): Loader for your test data
        tokenizer (Tokenizer): Tokenizer of model
        device (Optional[str], optional): Defaults to 'cuda'.
    """
    corrector: Corrector = Corrector()
    
    
    def __init__(
        self, 
        model: torch.Module, 
        loader: torch.utils.data.Dataloader, 
        tokenizer: Tokenizer, 
        device: Optional[str] = 'cuda'
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.loader = loader
        self.device = torch.device(device)
        
        self.CER = CharErrorRate()
        self.WER = WordErrorRate()
        
        # Predictions without any correcting.
        # Its saved after the first evaluate run 
        self.original_pred = []
        self.original_labels = []
        
        # In order not to count stat every time
        # we just save it to this dict
        self.avg_matches = 0
        self.symbol_err = {}
        self.length_word_CER = {}
    
    
    def evaluate(
        self,
        beam_width: Optional[int] = 0,
        correcting: Optional[bool] = False
    ) -> Tuple[float, float]:
        """
        This method evaluates model by CER and WER

        Args:
            beam_width (Optional[int]): If you want to use Beam Search Decoding Algorithm set it to 0.
                Defaults to 0.
            correcting (Optional[bool]): If you want to use correcting by dict. Defaults to False.

        Returns:
            Tuple[float, float]: CER, WER
        """
        
        predictions, labels = self._forward(beam_width)

        # Correct predictions if wants
        if correcting: 
            predictions = self.corrector.word_correction(predictions)
        
        # Count CER, WER
        char_error = self.CER(predictions, labels)
        word_error = self.WER(predictions, labels)
        
        return char_error, word_error
    
    
    def errors_sym_stat(self) -> Tuple[Dict[str, Dict], Dict[int, float]]:
        """
        This method calculate stat from collect_statistics method

        Returns:
            Tuple[Dict[str, Dict], Dict[str, float]]: First element - mismatched symbols,
            Second element - length-depended CER
        """
    
        if not len(self.length_word_CER):
            self._collect_statistics()
            self.avg_matches /= len(self.symbol_err.keys())
        
            # For every error symbol leavy only with number of errors >= self.avg_matches
            for pred_sym in self.symbol_err.keys():
                self.symbol_err[pred_sym] = dict(filter(
                    lambda elem: elem[1] >= self.avg_matches, 
                    self.symbol_err[pred_sym].items()))
                
            # Delete empty dictionaries in final stat
            self.symbol_err = dict(filter(
                lambda elem: len(elem[1]) > 0,
                self.symbol_err.items()))

            # Count mean value for CER based on length
            self.length_word_CER = {
                key : torch.Tensor(self.length_word_CER[key]).mean().item()
                for key in self.length_word_CER.keys()
                }
        
        return self.symbol_err.copy(), self.length_word_CER.copy()
    
    
    def _collect_statistics(self) -> None:
        """
        This method collect stat about mismatched symbols pairs
        and CERs for every phrase length
        """
        
        # If there's no evaluating stage
        if not len(self.original_pred):
            self.evaluate()
        
        for pred, true in zip(self.original_pred, self.original_labels):
            
            # Add CERs for all pairs (pred, true) to collect
            # the errors dependence on the length
            if len(true) in self.length_word_CER.keys():
                self.length_word_CER[len(true)].append(self.CER(pred, true))
            else: 
                self.length_word_CER[len(true)] = [self.CER(pred, true)]
            
            # Collect pairs of mismatched symbols
            if len(true) == len(pred) and true != pred:
                for i, j in zip(pred, true):
                    if i != j: 
                        if i in self.symbol_err.keys(): 
                            if j in self.symbol_err[i].keys():
                                self.symbol_err[i][j] += 1
                                self.avg_matches += 1
                            else: self.symbol_err[i][j] = 1
                        else: self.symbol_err[i] = {j : 1}
    
    
    def _forward(self, beam_width: int) -> Tuple[List[str], List[str]]:
        """
        This method apply model to loader and decoding probs to phrases

        Args:
            beam_width (int): If you dont want to use Beam Search Decoding Algorithm set it to 0

        Returns:
            Tuple[List[str], List[str]]: First element - predicted phrases,
            Second element - true phrases
        """
        
        predictions = []
        labels = []
        
        for batch in tqdm(self.loader):
            data, targets = batch[0].to(self.device), batch[1]
            labels.extend(targets)
            
            logits = self.model(data).contiguous().detach()
            L, N, H_out = logits.size() # L - seq length, N - batch size, H_out - len alphabet
            
            if beam_width:
                sim_preds = self.tokenizer.beam_decode(logits, N, beam_width)
            else:
                pred_sizes = torch.LongTensor([L for i in range(N)])
                probs, pos = logits.max(2)
                pos = pos.transpose(1, 0).contiguous().view(-1)
                sim_preds = self.tokenizer.decode(pos.data, pred_sizes.data)
                
            predictions.extend(sim_preds)
        
        if not len(self.original_labels):
            self.original_pred = predictions
            self.original_labels = labels
            
        return predictions, labels