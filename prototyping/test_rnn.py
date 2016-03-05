from __future__ import print_function
import PyTorch

PyTorch.require('rnn')

from PyTorchAug import nn

lstm = nn.LSTM(3,4)
print('lstm', lstm)

