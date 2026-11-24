import numpy as np
import matplotlib.pyplot as plt
import math
import random
import tensor as t
#figure this out later !!!!!

#building a fully connected layer with multiple neurons

class layer:
    def __init__(self,nin,nout,act=True):
        # scale by sqrt(3/nin) so outputs have variance ~1 and tanh doesn't saturate
        self.w = t.tensor(np.random.uniform(-1,1,(nin,nout)) * np.sqrt(3 / nin)) #weights of the whole layer
        self.b = t.tensor(np.zeros((nout,))) #biases of the whole layer
        self.act = act #False for the last layer, so it outputs raw logits

    def __call__(self,x): #forward pass for the whole layer
        x = x if isinstance(x, t.tensor) else t.tensor(x)
        out = x @ self.w + self.b
        return out.tanh() if self.act else out

    def parameters(self):
        return [self.w, self.b]


class MLP:
    def __init__(self,nin,nouts):
        sz= [nin] + nouts
        # every layer uses tanh except the last, which returns logits for cross-entropy
        self.Layers = [layer(sz[i],sz[i+1], act=(i < len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self,x):
        for l in self.Layers:
            x = l(x)
        return x
    def parameters(self):
        return [p for l in self.Layers for p in l.parameters() ]




    





