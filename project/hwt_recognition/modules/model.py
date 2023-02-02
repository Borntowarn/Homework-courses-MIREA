import torch
import torch.nn as nn

from typing import Union, Optional


class RNN(nn.Module):
    """
    This class use for seq 2 seq prediction

    Args:
        in_len (int): Input length of one timestamp. Number of input params
        out_len (int): Output length of one timestamp. Number of hidden neurons of RNN.
        n_classes (Optional[int], optional): If you want to use linear layer to final predict after RNN.
            n_classes is length of alphabet in your model. Defaults to None.
        rnn_type (Optional[str], optional): 'LSTM' or 'GRU' or 'RNN'. Defaults to 'RNN'.
        bidirectional (Optional[bool], optional): If True, becomes a bidirectional RNN. 
            out_len is multiplied by 2 Defaults to True.
        batch_first (Optional[bool], optional): If True, then the input and output tensors are provided
            as (batch, seq, feature) instead of (seq, batch, feature). Defaults to True.
    """

    
    def __init__(
        self,
        in_len: int,
        out_len: int,
        n_classes: Optional[int] = None,
        rnn_type: Optional[str] = 'RNN',
        bidirectional: Optional[bool] = True,
        batch_first: Optional[bool] = True
    ) -> None:
        super(RNN, self).__init__()
        self.n_classes = n_classes
        self.linear = None
        
        self.rnn = getattr(nn, rnn_type)(in_len, out_len, 
                          bidirectional = bidirectional, 
                          batch_first = batch_first)
        
        if self.n_classes:
            self.linear = nn.Linear(out_len * [1, 2][bidirectional], n_classes)

    
    def forward(self, data: torch.Tensor) -> torch.Tensor:        
        N, L, in_len = data.shape
        
        """
        N = batch size
        L = sequence length
        D = 2 if bidirectional=True otherwise 1
        in_len = input_size
        out_len = hidden_size
        """
        
        data, _ = self.rnn(data) # [N, L, D * out_len]
        if self.linear:
            data = data.reshape(N * L, -1)
            data = self.linear(data)
            data = data.reshape(N, L, -1)
        
        return data


class Model(nn.Module):
    """
    This class use for slicing initial image

    Args:
        in_channels (int): Number of channels of input image
        img_shape (tuple): Shape of input image
        len_alphabet (int): Length of training alphabet
        modules_seq (Optional[str], optional): Sequence of applying modules.
            C is Conv2d, A is Activation, M is MaxPool2d B is BatchNorm2d. Defaults to 'CAMB'.
        modules_freq (Optional[list], optional): Frequency of applying every module.
            Value 1 apply module every layer, 2 every 2 layers. Defaults to [1, 1, 1, 1].
        num_layers (Optional[int], optional): Number of modules_seq. Defaults to 5.
        increase_channels_layers (Optional[list], optional): After which layer the number
            of channels must be increased. Defaults to [1, 2, 4].
        conv_kernel_size (Optional[Union[int, tuple]], optional): Kernel size in Conv2d. Defaults to 3.
        conv_stride (Optional[Union[int, tuple]], optional): Stride in Conv2d. Defaults to 1.
        conv_padding (Optional[Union[int, tuple]], optional): Padding in Conv2d. Defaults to 1.
        pool_kernel_size (Optional[Union[int, tuple]], optional): Kernel size in MaxPool2d. Defaults to 2.
        pool_stride (Optional[Union[int, tuple]], optional): Stride in MaxPool2d. Defaults to 2.
        pool_padding (Optional[Union[int, tuple]], optional): Padding in MaxPool2d. Defaults to 0.
        activation (Optional[str], optional): Type of activation function. Defaults to 'ReLU'.
        rnn_type (Optional[str], optional): Type of RNN - 'RNN', 'GRU' or 'LSTM'. Defaults to 'RNN'.
    """
    
    def __init__(
        self,
        in_channels: int,
        img_shape: tuple,
        len_alphabet: int,
        modules_seq: Optional[str] = 'CAMB',
        modules_freq: Optional[list] = [1, 1, 1, 1],
        num_layers: Optional[int] = 5,
        increase_channels_layers: Optional[list] = [1, 2, 4],
        conv_kernel_size: Optional[Union[int, tuple]] = 3,
        conv_stride: Optional[Union[int, tuple]] = 1,
        conv_padding: Optional[Union[int, tuple]] = 1,
        pool_kernel_size: Optional[Union[int, tuple]] = 2,
        pool_stride: Optional[Union[int, tuple]] = 2,
        pool_padding: Optional[Union[int, tuple]] = 0,
        activation: Optional[str] = 'ReLU',
        rnn_type: Optional[str] = 'RNN'
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
            
            # Increasing number of channels on next layer
            if layer in increase_channels_layers:
                out_channels *= 2
        
        self.rnn_layers = nn.Sequential(
            RNN(img_shape[0]*512, 512, rnn_type=rnn_type, bidirectional=True, batch_first=True),
            RNN(1024, 128, len_alphabet + 1, rnn_type=rnn_type, bidirectional=True, batch_first=True)
        )
        
        print(f'Shape after convs layers: {img_shape}')
    
    
    def conv_output_shape(
        self,
        shape : tuple,
        kernel_size : Union[int, tuple] = 1,
        stride : Union[int, tuple] = 1,
        pad : Union[int, tuple] = 0,
        dilation : Union[int, tuple] = 1
    ) -> tuple:
        """
        This method calculate out height and width of img

        Args:
            shape (tuple): Input shape.
            kernel_size (Union[int, tuple], optional): Defaults to 1.
            stride (Union[int, tuple], optional): Defaults to 1.
            pad (Union[int, tuple], optional): Defaults to 0.
            dilation (Union[int, tuple], optional): Defaults to 1.

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
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        h = floor(((shape[0] + (2 * pad[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
        w = floor(((shape[1] + (2 * pad[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
        return (h, w)
    
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        
        for module in self.layers:
            data = module(data)
        
        bs, c, h, w = data.shape
        
        """
        N = batch size
        L = sequence length
        H_in = input_size
        H_out = len_alphabet
        """
        
        data = data.permute(0, 3, 1, 2).reshape(bs, w, c * h) # Out - bs, w, h * c (N, L, H_in)
        data = self.rnn_layers(data) # Out - N, L, H_out
        data = data.permute(1, 0, 2) # Out - L, N, H_out
        
        # For every img in batch Calc log_softmax on every timestamp
        prob = torch.nn.functional.log_softmax(data, 2) # Out - L, N, H_out
        
        return prob