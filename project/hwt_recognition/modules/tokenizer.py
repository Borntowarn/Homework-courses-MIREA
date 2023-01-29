import torch

from ctc_decoder import beam_search
from typing import Union, Iterable, Tuple

class Tokenizer:
    """
    Class needs to encode initial phrases to Tensor
    and decode predicted labels to phrases
    """
    
    def __init__(self, alphabet: Iterable[str]) -> None:
        
        self.alphabet = ''.join(sorted(alphabet))
        self.sym2class, self.class2sym = {'' : 0}, {0 : ''}
        
        for num, alpha in enumerate(self.alphabet):
            self.sym2class[alpha] = num + 1 # symbol to label 
            self.class2sym[num + 1] = alpha # label to symbol
    
    
    def encode(self, text: list[str]) -> Tuple[torch.IntTensor, torch.IntTensor]:
        """
        This method encode initial phrases to Tensor

        Args:
            text (list[str]): Initial phrases for encode

        Returns:
            tuple: First value is a tensor of phrases labels, second is lengths of phrases
        """
        
        length = []
        result = []
        
        for phrase in text:
            length.append(len(phrase))
            for alpha in phrase:
                if alpha in self.alphabet: 
                    result.append(self.sym2class[alpha])
                else: result.append(0)
        
        return (torch.IntTensor(result), torch.IntTensor(length))
    
    
    def decode(self, text: torch.Tensor, length: torch.IntTensor) -> Union[str, list]:
        """
        This method used for decoding prediction labels to text

        Args:
            text (torch.Tensor): predicted labels of symbols
            length (torch.IntTensor): lengths of prediction phrases

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
    
    
    def beam_decode(self, logits: torch.Tensor, batch_num: int, beam_width: int = 5) -> list[str]:
        """
        This method apply Beam Decoding Algorithm for model output

        Args:
            logits (torch.Tensor): Model output with log_softmax
            batch_num (int): Number of images in batch
            beam_width (int, optional): Beam width in algorithm. Defaults to 5.

        Returns:
            list[str]: Decoded phrases
        """
        predictions = []
        for i in range(batch_num):
            word = torch.nn.functional.softmax(logits[:, i, :], dim = 1)
            word = torch.hstack((word, word[:, 0].unsqueeze(1)))[:, 1:].cpu().numpy()
            res = beam_search(word, self.alphabet, beam_width) # Over 10 is too slow
            predictions.append(res)
        return predictions
    
    
    def __len__(self):
        return len(self.class2sym)