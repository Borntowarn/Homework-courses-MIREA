import torch
import enchant

from tqdm.notebook import tqdm
from torchmetrics import CharErrorRate, WordErrorRate

class Evaluator:
    """
    Class for evaluate CER, WER of model and
    count stat about symbols errors.
    """
    
    def __init__(self,
                 model,
                 loader,
                 coder,
                 device : str = 'cuda'
                 ) -> None:
        self.model = model.eval()
        self.CER = CharErrorRate()
        self.WER = WordErrorRate()
        self.coder = coder
        self.loader = loader
        self.device = device
        self.avg_matches = 0
        
        self.original_pred = []
        self.original_labels = []
        
        self.symbol_err = {}
        self.length_word_CER = {}
        
    
    def suggest(self, words: list, dictionary: enchant.Dict) -> str:
        result = ''
        
        for word in words:
            if word.isalpha():
                
                # If word is in dict probably it's without errors
                cer_suggest = dict()
                if dictionary.check(word):
                    result += word + ' '
                    continue
                
                # Else dict can suggest what word we need
                suggestions = set(dictionary.suggest(word))

                # For every suggestion finding CER
                for suggest in suggestions:
                    if ' ' not in suggest:
                        cer = self.CER(suggest, word)
                        cer_suggest[cer] = suggest
                
                # Get the nearest word
                if len(cer_suggest.keys()) > 0: 
                    result += cer_suggest[min(cer_suggest.keys())] + ' '
                # Or take original word if there's no suggestions
                else:
                    result += word + ' '
                    
        return result[:-1]

    
    def word_correction(self, predictions) -> list:
        correct_predictions = []
        dictionary = enchant.Dict("ru_RU")
        
        # Every pred phrase is splitted by words
        # then every word is checked with external dict
        for phrase in tqdm(predictions, total=len(predictions)):
            words = phrase.split()
            result = self.suggest(words, dictionary)
            correct_predictions.append(result)
            
        return correct_predictions
    
    
    def count_errors(self) -> None:
        
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
    
    
    def errors_sym_stat(self) -> tuple[dict[str, dict], dict[str, float]]:
    
        if not len(self.length_word_CER):
            self.count_errors()
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
    
    
    def forward(self, beam_search: bool) -> tuple[list[str], list[str]]:
        
        predictions = []
        labels = []
        
        for iteration, batch in enumerate(tqdm(self.loader)):
            data, targets = batch[0].to(self.device), batch[1]
            labels.extend(targets)
            
            logits = self.model(data).contiguous().detach()
            T, B, H = logits.size()
            pred_sizes = torch.LongTensor([T for i in range(B)])
            probs, pos = logits.max(2)
            pos = pos.transpose(1, 0).contiguous().view(-1)
            
            if beam_search:
                sim_preds = self.coder.beam_decode(logits, B)
            else:
                sim_preds = self.coder.decode(pos.data, pred_sizes.data)
                
            predictions.extend(sim_preds)
        
        if not len(self.original_labels):
            self.original_pred = predictions
            self.original_labels = labels
            
        return predictions, labels

    def evaluate(self, beam_search: bool = False, correcting: bool = False) -> tuple:
        
        predictions, labels = self.forward(beam_search)
        
        # Correct predictions if wants
        if correcting: 
            predictions = self.word_correction(predictions)
        
        # Count CER, WER
        char_error = self.CER(predictions, labels)
        word_error = self.WER(predictions, labels)
        
        return char_error, word_error