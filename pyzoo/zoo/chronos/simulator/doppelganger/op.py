import torch.nn as nn
import numpy as np


class linear(nn.Module):
    '''
    This is an nn.Module implementation to op.linear.
    The difference between this implementation and
    torch.nn.linear is that this implementation take input
    shape (batch_size, *) and output shape is (batch_size, output_size)
    '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_torch = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.linear_torch(x)
        return x

def flatten(x):
    '''
    This is an torch implementation to op.flatten.
    '''
    return x.view(-1, np.prod(x.shape[1:]))

# op.batch_norm will be replaced by torch.nn.functional.batch_norm
# op.layer_norm will be replaced by torch.nn.functional.layer_norm
# op.deconv2d and op.conv2d is not used in any original tf code
# op.lrelu will be replaced by torch.nn.LeakyReLU
