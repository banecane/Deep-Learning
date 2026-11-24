import numpy as np
import matplotlib.pyplot as plt
import math





#unbroadcasting numpy arrays to calculate backward pass
def unbroadcast(grad, shape):
    """Sum grad down to `shape`, reversing NumPy broadcasting."""
    # 1. sum away extra leading dims that broadcasting added
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2. sum over dims that were size 1 in the original
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
#tensor for a cnn network where each neuron will be given by this one tensor
class tensor:
    def __init__(self, data, _children = (), _o= ''):
        self.data = np.asarray(data, dtype= np.float64) #data can be numpy array this time
        self._prev = set(_children)
        self.op = _o
        self._backward = lambda: None
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"tensor(data={self.data})"

    #functions

    def __add__(self,other):
        other = other if isinstance(other, tensor) else tensor(other)
        out = tensor(self.data + other.data, (self,other), '+')
        def _backward():
            self.grad += unbroadcast(out.grad,self.data.shape)
            other.grad += unbroadcast(out.grad,other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other, tensor) else tensor(other)
        out = tensor( self.data * other.data, (self,other),'*')
        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad+= unbroadcast(self.data * out.grad , other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self,other):
        other = other if isinstance(other, tensor) else tensor(other)
        out = tensor( self.data @ other.data, (self,other),'@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self *-1

    def __sub__(self,other):
        return self + (-other)

    def __rsub__(self, other):      # other - self   (e.g. 3 - v)
        return other + (-self)

    def __radd__(self, other):      # other + self   (e.g. 3 + v)
        return self + other


    def __rmul__(self, other):      # other * self   (e.g. 3 * v)
        return self * other
    def __rtruediv__(self,other):
        return other*self** -1

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"only supporting int/float powers for now"
        out = tensor(self.data**other,(self,))
        def _backward():
            self.grad += unbroadcast(other*(self.data**( other -1))*out.grad,self.data.shape)
        out._backward = _backward
        return out           

    def cross_entropy_loss(self, ycorrect):   # ycorrect: index of the correct class
        z = self.data.reshape(-1)              # works for (10,) or (1, 10)
        e = np.exp(z - z.max())                # stable softmax
        p = e / e.sum()

        out = tensor(np.array([-np.log(p[ycorrect])]), (self,), 'ce')

        def _backward():
            g = p.copy()
            g[ycorrect] -= 1                   # p - one_hot
            self.grad += g.reshape(self.data.shape) * out.grad
        out._backward = _backward
        return out

        


    def tanh(self):
        n = self.data
        t = (np.exp(2*n) -1)/(np.exp(2*n) + 1)
        def _backward():
            self.grad += unbroadcast((1 - t**2)*out.grad,self.data.shape)
        out = tensor( t, (self,))
        out._backward = _backward
        return out
    def sum(self):
        t = sum(self.data)
        def _backward():
            self.grad = np.ones_like(self.grad)*out.grad
        out = tensor(t,(self,))
        out._backward = _backward
        return out


    def reshape(self, *shape):
        out = tensor(self.data.reshape(*shape), (self,))
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out
    def flatten(self):
        return self.reshape((1,-1))

    #for the convolutional layer
    def convolve(self,other,k,s,oh,ow):
        #assuming that self.size < other.size
        #write self,other as _children but in the backward function only affect the grad of a particular patch
        other = other if isinstance(other, tensor) else tensor(other)
        tt = np.zeros((oh,ow))
        for i in range(oh):
            for j in range(ow):
                r,t = i*s, j*s
                patch = other.data[r:r+k,t:t+k,:]
                tt[i,j] = np.sum(self.data * patch)

        def _backward():
            for i in range(oh):
                for j in range(ow):
                    r,t = i*s,j*s
                    self.grad+= out.grad[i,j]*other.data[r:r+k,t:t+k,:]
                    other.grad[r:r+k,t:t+k,:] += out.grad[i,j]*self.data


        out = tensor(tt,(self,other))
        out._backward = _backward
        return out



    #for the pooling layer
    def maxPool(self, kernelSize, stride):
        k, s = kernelSize, stride
        x = self.data if self.data.ndim == 3 else self.data[:, :, None]
        H, W, C = x.shape
        oh = (H - k) // s + 1
        ow = (W - k) // s + 1

        out_data = np.zeros((oh, ow, C))
        arg = np.zeros((oh, ow, C), dtype=int)   # flat index inside each window, per channel
        ch = np.arange(C)

        for i in range(oh):
            for j in range(ow):
                r, t = i * s, j * s
                win = x[r:r+k, t:t+k, :].reshape(k * k, C)
                idx = win.argmax(axis=0)          # one winner per channel
                arg[i, j] = idx
                out_data[i, j] = win[idx, ch]

        if self.data.ndim == 2:
            out_data = out_data[:, :, 0]
        out = tensor(out_data, (self,))


        def _backward():
            g = out.grad if out.grad.ndim == 3 else out.grad[:, :, None]
            gx = np.zeros_like(x)
            for i in range(oh):
                for j in range(ow):
                    di, dj = np.divmod(arg[i, j], k)   # window-local (row, col)
                    gx[i*s + di, j*s + dj, ch] += g[i, j]
            self.grad += gx.reshape(self.data.shape)

        out._backward = _backward
        return out



    @staticmethod
    def stack_channels(tensors):
        arr = [t.data if t.data.ndim ==3 else t.data[:,:,None] for t in tensors]
        out = np.concatenate(arr,axis = -1)
        def _backward():
            k = 0
            for i in range(len(arr)):
                tensors[i].grad += out.grad[:,:,k: k+ arr[i].shape[2]].reshape(tensors[i].data.shape)
                
                k+= arr[i].shape[2]
        out = tensor(out,tuple(tensors))
        out._backward = _backward
        return out



    #Backpropogation
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
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
