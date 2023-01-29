import enchant

from typing import List
from tqdm.notebook import tqdm
from torchmetrics import CharErrorRate

class Corrector:
    """
    Class for correcting predicted phrases based on external dictionary
    """
    
    def __init__(self) -> None:
        self.CER = CharErrorRate()
        self.dictionary = enchant.Dict("ru_RU")

    
    def word_correction(self, predictions: List[str]) -> List[str]:
        """
        This method correct predicted phrases using suggest method

        Args:
            predictions (list[str]): Initial decoded predicted phrases

        Returns:
            List[str]: Corrected phrases
        """
        correct_predictions = []
        
        # Every pred phrase is splitted by words
        # then every word is checked with external dict
        for phrase in tqdm(predictions, total=len(predictions)):
            words = phrase.split()
            result = self._suggest(words)
            correct_predictions.append(result)
            
        return correct_predictions
    
    
    def _suggest(self, words: List[str]) -> str:
        """
        This method suggest a word based on external dictionary

        Args:
            words (List[str]): Words in phrase
            dictionary (enchant.Dict): External dictionary provided by enchant module

        Returns:
            str: Final phrase
        """
        result = ''
        
        for word in words:
            if word.isalpha():
                
                # If word is in dict probably it's without any errors
                cer_suggest = dict()
                if self.dictionary.check(word):
                    result += word + ' '
                    continue
                
                # Else dict can suggest what word we need
                suggestions = set(self.dictionary.suggest(word))

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


if __name__ == '__main__':
    corrector = Corrector()
    print(corrector.word_correction(['Контаиты']))