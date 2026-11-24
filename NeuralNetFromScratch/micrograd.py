#following andrey karpathy's tutorial for micrograd


import math
import numpy as np
import matplotlib.pyplot as plt
import torch 
import random
import graphviz


class Value:
    def __init__(self,data, _children = (),_op = ' ' ):
        self.data = data #internal weight of the model
        self.grad = 0.0 #denotes the derivative of output with respect to this variable/value
        self._backward = lambda: None
        self._prev = set(_children)
        self.op = _op
    def __repr__(self):
        return f"Value(data={self.data})"

    #functions
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other,Value) else  Value(other)
        out = Value( self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self*-1

    def __sub__(self,other):
        return self + (-other)

    def __rsub__(self, other):      # other - self   (e.g. 3 - v)
        return other + (-self)

    def __radd__(self, other):      # other + self   (e.g. 3 + v)
        return self + other


    def __rmul__(self, other):      # other * self   (e.g. 3 * v)
        return self * other

    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"only supporting int/float powers for now"
        out = Value(self.data**other,(self,))
        def _backward():
            self.grad += other*(self.data**( other -1))*out.grad
        out._backward = _backward
        return out

    def __truediv__(self,other):
        return self* other**-1

    def tanh(self):
        n = self.data
        t = (math.exp(2*n) -1)/(math.exp(2*n) + 1)
        def _backward():
            self.grad += (1 - t**2)*out.grad
        out = Value( t, (self,))
        out._backward = _backward
        return out



    #back-propoagation
    def topSort(self):#dfs approach, maxes out at a depth of 1000
        L =[]
        V = set()
        def sort(x):
            if x not in V:
                V.add(x)
                for m in x._prev:
                    sort(m)
                L.append(x)
        sort(self)
        return L
    def backward(self):
        topo = self.topSort()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()














#building a neural net now
class Neuron:
    def __init__(self, nin):
        self.w = [ Value(random.uniform(-1,1)) for _ in range(nin) ]
        self.b = Value(random.uniform(-1,1))
    def __call__(self,x):#calculate the forward pass for a single neuron
            act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
            out = act.tanh()
            return out
    def parameters(self):
        return self.w + [self.b]
class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):#forward pass for the single layer
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    def parameters(self):
        return [p for neurons in self.neurons for p in neurons.parameters()]

class MLP:
    def __init__(self,nin,nouts):#nouts is a list containing the size of all the layers
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):#forward pass through the nn
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters() ]



if __name__ == "__main__":
    # gradient checks, quick tests, etc.
    np.random.seed(0)
    nin = 3
    nouts = [4,4,1]
    n = MLP(nin, nouts)

    xs = [[2,3,-1],
          [3,-1,0.5],
        [0.5,1,1],
        [1,1,-1]]
    ys= [1,-1,-1,1]#desired targets
    ypred = [n(x) for x in xs]


    for k in range(50):
        # 1. forward pass: must run again every iteration
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # 2. zero grads, then backward
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # 3. update
        for p in n.parameters():
            p.data += -0.05 * p.grad

        print(k, loss.data)





    for x in xs: print(n(x))











