import numpy as np
import sys
import os

class CIFAR10(object):
    def __init__(self):
        self.mean = [125.307, 122.95, 113.865]
        self.std = [62.9932, 62.0887, 66.7048]

    def preprocess(self, x_in):
        if len(x_in.shape) != 3 and len(x_in.shape) != 4:
            print('error shape', x_in.shape)
            sys.exit()
        x_in = x_in.astype('float32')
        if len(x_in.shape) == 3:
            for i in range(3):
                x_in[:, :, i] = (x_in[:, :, i] - self.mean[i]) / self.std[i]
        elif len(x_in.shape) == 4:
            for i in range(3):
                x_in[:, :, :, i] = (x_in[:, :, :, i] - self.mean[i]) / self.std[i]
        return x_in
    
    
    def deprocess(self, x_in):
        if len(x_in.shape) != 3 and len(x_in.shape) != 4:
            print('error shape', x_in.shape)
            sys.exit()
        x_in = x_in.astype('float32')
        if len(x_in.shape) == 3:
            for i in range(3):
                x_in[:, :, i] = x_in[:, :, i] * self.std[i] + self.mean[i]
        elif len(x_in.shape) == 4:
            for i in range(3):
                x_in[:, :, :, i] = x_in[:, :, :, i] * self.std[i] + self.mean[i]
        return x_in
