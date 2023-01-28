import PIL
import torch

from tkinter import *
from PIL import ImageDraw
from Evaluator import Evaluator
from Transforms import Transforms

class Recognizer:
    """
    This class can recognize phrase from painted or given picture
    """
    
    #model: Model
    device: torch.device
    
    def __init__(self,
                 model,
                 coder,
                 transform: Transforms,
                 device: str = 'cuda'
                 ) -> None:
        self.model = model.eval()
        self.transform = transform
        self.device = device
        self.coder = coder
    
    
    def forward(self, img: Image, beam_width: int) -> str:
        """
        This method implements forward pass of the model

        Args:
            img (Image): Given image to recognize.
            beam_width (int): If > 0 beam decoding will be appplied
            If Image then transforms must be True. Else it's assumed transforms have already been applied

        Returns:
            str: Recognized phrase
        """
        img = self.transform(img).unsqueeze(0)

        logits = self.model(img.to(self.device))
        logits = logits.contiguous().cpu()
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)
        if beam_width:
            prediction = self.coder.beam_decode(logits, B, beam_width)
        else:
            prediction = self.coder.decode(pos.data, pred_sizes.data)
        
        return prediction[0]
    
    
    def paint(self) -> None:
        """
        This method creates a window to paint a phrase
        """
        width = 1000  # canvas width
        height = 400 # canvas height
        center = height//2
        white = (255, 255, 255) # canvas back
        
        self.master = Tk()

        # Create a tkinter canvas to draw on
        self.canvas = Canvas(self.master, width=width, height=height, bg='white')
        self.canvas.pack()

        # Create an empty PIL image and draw object to draw on
        self.img = PIL.Image.new("RGB", (width, height), white)
        self.draw = ImageDraw.Draw(self.img)
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.draw_img)

        # Button to recognize img and close pint window
        button=Button(text="Recognize",command=self.master.destroy)
        button.pack()
        
        self.master.mainloop()
    

    def draw_img(self, event) -> None:
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black",width=5)
        self.draw.line([x1, y1, x2, y2],fill="black",width=5)


    def recognize_from_painted(self, beam_width: int = 0) -> str:
        """
        This method allows you to paint phrase to recognize
        
        Args:
            beam_width (int): If > 0 beam decoding will be appplied. Default 5.

        Returns:
            str: Recognized phrase
        """
        self.paint()
        prediction = self.forward(self.img, beam_width)
        return prediction

    
    def recognize_from_file(self, path, beam_width: int = 0, correcting = False) -> str:
        """
        This method loads img from given then recognize it

        Args:
            path (str): Path to img
            beam_width (int): If > 0 beam decoding will be appplied. Default 5.
            correcting (bool, optional): If you want to correct predicted word with dictionary.
            Defaults to False.

        Returns:
            str: Recognized phrase
        """
        img = PIL.Image.open(path)
        prediction = self.forward(img, beam_width)
        
        if correcting: 
            evaluator = Evaluator(self.model, None, self.coder)
            prediction = evaluator.word_correction([prediction])[0]
            
        return prediction