from __future__ import print_function
import PyTorch
from PyTorchAug import nn
PyTorch.require('rnn')


if __name__ == '__main__':
    lstm = nn.LSTM(3, 4)
    print('lstm', lstm)
