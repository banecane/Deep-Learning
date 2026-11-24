import numpy as np
from micrograd import Value
from cnn import tensor 
x = Value(3)
y= Value(4)
z = x+y*3

xx = tensor(3)
yy = tensor(4)
zz = xx+yy


