import torch
import torch.nn as nn

from torch import Tensor
from typing import Union


class RNN(nn.Module):
    """
    This class use for seq 2 seq prediction
    """
    
    def __init__(self,
                 in_len,
                 out_len,
                 n_classes: int = None,
                 rnn_type: str = 'RNN',
                 bidirectional: bool = True,
                 batch_first: bool = True) -> None:
        
        super(RNN, self).__init__()
        self.n_classes = n_classes
        self.linear = None
        
        self.rnn = getattr(nn, rnn_type)(in_len, out_len, 
                          bidirectional = bidirectional, 
                          batch_first = batch_first)
        
        if self.n_classes:
            self.linear = nn.Linear(out_len * [1, 2][bidirectional], n_classes)

    
    def forward(self, data) -> torch.Tensor:
        
        """
        N = batch size
        L = sequence length
        D = 2 if bidirectional=True otherwise 1
        H_in = input_size
        H_out = hidden_size
        """
        
        N, L, H_in = data.shape
        
        data, _ = self.rnn(data) # [N, L, D * n_hidden]
        if self.linear:
            data = data.reshape(N * L, -1)
            data = self.linear(data)
            data = data.reshape(N, L, -1)
        
        return data


class Model(nn.Module):
    """
    This class use for slicing initial image
    """
    
    def __init__(self,
                 in_channels: int,
                 img_shape: tuple,
                 len_alphabet: int,
                 num_layers: int = 5,
                 increase_channels_layers: list = [1, 2, 4],
                 modules_seq: str = 'CAMB',
                 modules_freq: list = [1, 1, 1, 1],
                 conv_kernel_size: Union[int, tuple] = 3,
                 conv_stride: Union[int, tuple] = 1,
                 conv_padding: Union[int, tuple] = 1,
                 pool_kernel_size: Union[int, tuple] = 2,
                 pool_stride: Union[int, tuple] = 2,
                 pool_padding: Union[int, tuple] = 0,
                 activation: str = 'ReLU',
                 rnn_type: str = 'RNN'
                 ) -> None:
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        out_channels = 64
        
        # Frequence of append every module. For example:
        # if freq for MaxPool is 2 then MaxPool will be appended every 2 layer
        frequency = dict(zip(modules_seq, modules_freq)) 
        
        # For every layer create sequence of modules
        for layer in range(1, num_layers + 1):
            for module in modules_seq:
                if layer % frequency[module]: # Check freq of module
                    continue
                if module == 'C':
                    self.layers.append(nn.Conv2d(in_channels, out_channels, conv_kernel_size,
                                                conv_stride, conv_padding))
                    in_channels = out_channels
                    img_shape = self.conv_output_shape(img_shape, conv_kernel_size, conv_stride, conv_padding)
                elif module == 'A':
                    self.layers.append(getattr(nn, activation)())
                elif module == 'M':
                    self.layers.append(nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding))
                    img_shape = self.conv_output_shape(img_shape, pool_kernel_size, pool_stride, pool_padding)
                elif module == 'B':
                    self.layers.append(nn.BatchNorm2d(out_channels))
            
            if layer in increase_channels_layers:
                out_channels *= 2
        
        self.rnn_layers = nn.Sequential(
            RNN(img_shape[0]*512, 512, rnn_type=rnn_type, bidirectional=True, batch_first=True),
            RNN(1024, 128, len_alphabet + 1, rnn_type=rnn_type, bidirectional=True, batch_first=True)
        )
        
        print(f'Shape after convs layers: {img_shape}')
    
    
    def conv_output_shape(self,
                          h_w : tuple,
                          kernel_size : Union[int, tuple] = 1,
                          stride : Union[int, tuple] = 1,
                          pad : Union[int, tuple] = 0,
                          dilation : Union[int, tuple] = 1
                          ) -> tuple:
        """
        This method calculate out height and width of img

        Args:
            h_w (tuple): Input shape.
            kernel_size (Union[int, tuple], optional): Defaults to 1.
            stride (Union[int, tuple], optional): Defaults to 1.
            pad (Union[int, tuple], optional): Defaults to 0.
            dilation (Union[int, tuple], optional): _description_. Defaults to 1.

        Returns:
            tuple: Output shape
        """
        from math import floor
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(pad, int):
            pad = (pad, pad)
            
        h = floor(((h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
        w = floor(((h_w[1] + (2 * pad[0]) - (dilation * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
        return (h, w)
    
    
    def forward(self, data) -> Tensor:
        
        for module in self.layers:
            data = module(data)
        
        bs, c, h, w = data.shape
        
        data = data.permute(0, 3, 1, 2).reshape(bs, w, c * h) # Out - bs, w, h * c (N, L, H)
        data = self.rnn_layers(data) # Out - N, L, len_alphabet
        data = data.permute(1, 0, 2) # Out - L, N, len_alphabet
        
        prob = torch.nn.functional.log_softmax(data, 2)
        
        return prob