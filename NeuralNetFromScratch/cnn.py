import numpy as np
import matplotlib.pyplot as plt
import math
import tensor as t



class c_neuron:
    def __init__(self,nh,nw,kernelSize,channels,stride):
        self.f = kernelSize
        self.s = stride
        self.c = channels
        fan_in = kernelSize * kernelSize * channels
        # scale by sqrt(3/fan_in) so outputs have variance ~1 and tanh doesn't saturate
        self.w = t.tensor(np.random.uniform(-1,1,(kernelSize,kernelSize,channels)) * np.sqrt(3 / fan_in)) #filter weights
        self.oh = (nh - kernelSize)//stride + 1
        self.ow = (nw - kernelSize)//stride + 1
        self.b = t.tensor(np.zeros((1,)))


    def __call__(self,x): #x is the numpy array which denotes the input
        x = x if isinstance(x,t.tensor) else t.tensor(x)
        out = self.w.convolve(x,self.f,self.s,self.oh,self.ow) + self.b
        out = out.tanh()
        return out

    def parameters(self):
        
        return [self.w] + [self.b]


#defining a cnn layer
class c_layer:
    def __init__(self,nh,nw,nc,kernelNumber,stride,kernelSize,maxPoolKernelSize, maxPoolStride):
        self.f = kernelSize
        self.s = stride
        self.no = kernelNumber #number of output channels
        self.mxpk = maxPoolKernelSize
        self.mxps = maxPoolStride
        self.neurons = [ c_neuron(nh,nw,self.f,nc,stride) for _ in range(kernelNumber)]

    def __call__(self, x):
        x = x if isinstance(x, t.tensor) else t.tensor(x)
        arr = [n(x) for n in self.neurons]
        out =  t.tensor.stack_channels(arr)
        return out.maxPool(self.mxpk,self.mxps)

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters() ]


#define multiple convolution layers
#a layer for us will be one convolution layer followed by a pooling layer

def output_size(input_size, kernel_size, stride, padding=0,
                pool_kernel_size=0, pool_stride=0):
    size = (input_size - kernel_size + 2 * padding) // stride + 1
    if pool_kernel_size > 0 and pool_stride > 0:
        size = (size - pool_kernel_size) // pool_stride + 1
    return size

def output_shape(input_shape, kernel_number, kernel_size, stride,
                 pool_kernel_size=0, pool_stride=0, padding=0):
    h = output_size(input_shape[0], kernel_size, stride, padding, pool_kernel_size, pool_stride)
    w = output_size(input_shape[1], kernel_size, stride, padding, pool_kernel_size, pool_stride)
    if h < 1 or w < 1:
        raise ValueError(f"layer shrinks {input_shape} to {h}x{w}")
    return (h, w, kernel_number)          # channels = number of kernels


class c_mln:
    def __init__(self, input_shape, specs):
        # input_shape: (H, W, C)
        # specs: list of (kernelNumber, stride, kernelSize, poolKernelSize, poolStride)
        self.layers = []
        shape = input_shape
        for kn, s, k, pk, ps in specs:
            nh, nw, nc = shape
            self.layers.append(c_layer(nh, nw, nc, kn, s, k, pk, ps))
            shape = output_shape(shape, kn, k, s, pk, ps)
        self.out_shape = shape

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]














