import PIL
import torch

from .tokenizer import Tokenizer
from .corrector import Corrector

from tkinter import *
from PIL import ImageDraw
from typing import Optional
from torchvision import transforms

class Recognizer:
    """
    Class for 

    Args:
        model (torch.Module): Model for recognizing
        tokenizer (Tokenizer): Tokenizer that is used in model
        transform (transforms.Compose): Test transforms to transform recognized image
        device (str, optional): Device to run model. Defaults to 'cuda'.
    """
    corrector: Corrector = Corrector()
    
    def __init__(
        self,
        model: torch.Module,
        tokenizer: Tokenizer,
        transform: transforms.Compose,
        device: Optional[str] = 'cuda'
    ) -> None:
        self.model = model.eval()
        self.transform = transform
        self.tokenizer = tokenizer
        self.device = torch.device(device)
    
    
    def recognize_from_painted(self, beam_width: int = 0) -> str:
        """
        This method allows you to paint phrase to recognize
        
        Args:
            beam_width (int): If > 0 beam decoding will be appplied. Default 5.

        Returns:
            str: Recognized phrase
        """
        self._paint()
        prediction = self.forward(self.img, beam_width)
        return prediction

    
    def recognize_from_file(
        self, 
        path: str, 
        beam_width: Optional[int] = 0,
        correcting: Optional[bool] = False
    ) -> str:
        """
        This method loads img from given then recognize it

        Args:
            path (str): Path to img
            beam_width (Optional[int], optional): If > 0 beam decoding will be appplied. Default 5.
            correcting (Optional[bool], optional): If you want to correct predicted word with dictionary.
            Defaults to False.

        Returns:
            str: Recognized phrase
        """
        img = PIL.Image.open(path)
        prediction = self.forward(img, beam_width)
        
        if correcting: 
            prediction = self.corrector.word_correction([prediction])[0]
            
        return prediction
    
    
    def _forward(self, img: PIL.Image, beam_width: int) -> str:
        """
        This method implements forward pass of the model

        Args:
            img (PIL.Image): Given image to recognize.
            beam_width (int): If > 0 beam decoding will be appplied

        Returns:
            str: Recognized phrase
        """
        img = self.transform(img).unsqueeze(0)

        logits = self.model(img.to(self.device))
        logits = logits.contiguous().cpu()
        L, N, H_out = logits.size() # L - seq length, N - batch size, H_out - len alphabet
        
        if beam_width:
            prediction = self.tokenizer.beam_decode(logits, N, beam_width)
        else:
            pred_sizes = torch.LongTensor([L])
            probs, pos = logits.max(2)
            pos = pos.transpose(1, 0).contiguous().view(-1)
            prediction = self.tokenizer.decode(pos.data, pred_sizes.data)
        
        return prediction
    
    
    def _paint(self) -> None:
        """
        This method creates a window to paint a phrase
        """
        width = 1000  # canvas width
        height = 400 # canvas height
        white = (255, 255, 255) # canvas back
        
        self.master = Tk()

        # Create a tkinter canvas to draw on
        self.canvas = Canvas(self.master, width=width, height=height, bg='white')
        self.canvas.pack()

        # Create an empty PIL image and draw object to draw on
        self.img = PIL.Image.new("RGB", (width, height), white)
        self.draw = ImageDraw.Draw(self.img)
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self._draw_img)

        # Button to recognize img and close pint window
        button=Button(text="Recognize",command=self.master.destroy)
        button.pack()
        
        self.master.mainloop()
    

    def _draw_img(self, event) -> None:
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black",width=5)
        self.draw.line([x1, y1, x2, y2],fill="black",width=5)