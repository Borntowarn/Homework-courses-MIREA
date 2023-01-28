import torch

from torch import Tensor
from typing import Union
from ctc_decoder import beam_search

class SymbolCoder:
    """
    Class needs to encode initial phrases to Tensor
    and decode predicted labels to phrases
    """
    
    def __init__(self, alphabet) -> None:
        
        self.alphabet = ''.join(sorted(alphabet))
        self.sym2class, self.class2sym = {'' : 0}, {0 : ''}
        
        for num, alpha in enumerate(self.alphabet):
            self.sym2class[alpha] = num + 1
            self.class2sym[num + 1] = alpha
    
    
    def encode(self, text) -> tuple[Tensor, Tensor]:
        """
        This method encode initial phrases to Tensor

        Args:
            text (list): Initial phrases for encode

        Returns:
            tuple: First value is a tensor of phrases labels, second is lengths of phrases
        """
        
        length = []
        result = []
        
        for word in text:
            length.append(len(word))
            for alpha in word:
                if alpha in self.alphabet: 
                    result.append(self.sym2class[alpha])
                else: result.append(0)
        
        return (torch.tensor(result, dtype=torch.int64), torch.tensor(length, dtype=torch.int64))
    
    
    def decode(self, text, length) -> Union[str, list]:
        """
        This method used for decoding prediction labels to text

        Args:
            text (Tensor): predicted labels of symbols
            length (Tensor): lengths of prediction phrases

        Returns:
            Union[str, list]: list type returns when use for batch, for single word returns str
        """
        
        #For single word
        if length.numel() == 1:
            length = length[0]
            word = ''
            
            for i in range(length):
                if text[i] != 0 and not (i > 0 and text[i - 1] == text[i]):
                    word  += self.class2sym[text[i].item()]
            return word
        
        #For batch
        else:
            words = []
            index = 0
            
            for i in range(length.numel()):
                l = length[i]
                words.append(self.decode(text[index:index + l], torch.IntTensor([l])))
                index += l
            return words
    
    
    def beam_decode(self, logits, batch_num, beam_width = 5) -> list[str]:
        predictions = []
        for i in range(batch_num):
            word = torch.nn.functional.softmax(logits[:, i, :], dim = 1)
            word = torch.hstack((word, word[:, 0].unsqueeze(1)))[:, 1:].cpu().numpy()
            res = beam_search(word, self.alphabet, beam_width) # Over 10 is too slow
            predictions.append(res)
        return predictions
    
    
    def __len__(self):
        return len(self.class2sym)