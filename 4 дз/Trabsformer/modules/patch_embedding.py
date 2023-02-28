from torch import nn


class PatchEmbedding(nn.Module):
    """ 
    Image to Patch Embedding
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        d_model: int = 768
    ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.in_chans = in_chans
        self.img_size = img_size
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embeddings = nn.Conv2d(3, self.d_model, patch_size, patch_size)

    def forward(self, image):
        b, c, h, w = image.shape
        
        assert h == self.img_size and w == self.img_size, f'Image size must be {self.img_size}x{self.img_size}'
        assert c == self.in_chans, f'Image must have {self.in_chans} channels'
        
        patches = self.patch_embeddings(image).reshape(b, self.d_model, -1).transpose(1, 2)
        
        return patches