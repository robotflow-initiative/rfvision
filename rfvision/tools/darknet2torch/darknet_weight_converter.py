import numpy as np
import torch
import torch.nn as nn
from rflib.cnn import ConvModule


class WeightLoader:
    """ Load darknet weight files into pytorch layers """

    def __init__(self, filename):
        with open(filename, 'rb') as fp:
            self.header = np.fromfile(fp, count=3, dtype=np.int32).tolist()
            ver_num = self.header[0] * 100 + self.header[1] * 10 + self.header[2]
            print(f'Loading weight file: version {self.header[0]}.{self.header[1]}.{self.header[2]}')

            if ver_num <= 19:
                print(
                    'Weight file uses sizeof to compute variable size, which might lead to undefined behaviour. (choosing int=int32, float=float32)')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int32)[0])
            elif ver_num <= 29:
                print(
                    'Weight file uses sizeof to compute variable size, which might lead to undefined behaviour. (choosing int=int32, float=float32, size_t=int64)')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int64)[0])
            else:
                print(
                    'New weight file syntax! Loading of weights might not work properly. Please submit an issue with the weight file version number. [Run with DEBUG logging level]')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int64)[0])

            self.buf = np.fromfile(fp, dtype=np.float32)

        self.start = 0
        self.size = self.buf.size

    def load_layer(self, layer):
        """ Load weights for a layer from the weights file """
        if type(layer) == nn.Conv2d:
            self._load_conv(layer)
        elif type(layer) == ConvModule:
            self._load_convbatch(layer)
        elif type(layer) == nn.Linear:
            self._load_fc(layer)
        else:
            raise NotImplementedError(f'The layer you are trying to load is not supported [{type(layer)}]')

    def _load_conv(self, model):
        num_b = model.bias.numel()
        model.bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                              .view_as(model.bias.data))
        self.start += num_b

        num_w = model.weight.numel()
        model.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_w])
                                .view_as(model.weight.data))
        self.start += num_w

    def _load_convbatch(self, model):
        num_b = model.bn.bias.numel()
        model.bn.bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                                        .view_as(model.bn.bias.data))
        self.start += num_b
        model.bn.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                                          .view_as(model.bn.weight.data))
        self.start += num_b
        model.bn.running_mean.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                                           .view_as(model.bn.running_mean))
        self.start += num_b
        model.bn.running_var.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                                          .view_as(model.bn.running_var))
        self.start += num_b

        num_w = model.conv.weight.numel()
        model.conv.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_w])
                                          .view_as(model.conv.weight.data))
        self.start += num_w

    def _load_fc(self, model):
        num_b = model.bias.numel()
        model.bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_b])
                              .view_as(model.bias.data))
        self.start += num_b

        num_w = model.weight.numel()
        model.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start + num_w])
                                .view_as(model.weight.data))
        self.start += num_w
