import numpy as np
import matplotlib.pyplot as plt
import math 
import random
import cnn
import MLP
import os, pickle
import time


DATA_DIR  = r"C:\Users\nilak\OneDrive\Desktop\Programming\python\datasets\cifar-10-python\cifar-10-batches-py"


def load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X.astype(np.float64) / 255.0, np.array(d[b"labels"])

def load_cifar10(data_dir):
    parts = [load_batch(os.path.join(data_dir, f"data_batch_{i}")) for i in range(1, 6)]
    X_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)




#building lenet5

specs = [
    (6, 1, 5, 2, 2),  # C1: 6 filters of size 5x5
    (16, 1, 5, 2, 2), # C3: 16 filters of size 5x5 
]
n = cnn.c_mln((32,32,3), specs)

y = [n(x) for x in X_train[:10]]


yy = [y[i].flatten() for i in range(len(y))] #(400,) shape for each image after flatteing the output of the last layer of the cnn

#Now we can use the output of the cnn as input to a fully connected layer

fnn = MLP.MLP(400,[120,84,10]) #fully connected layer with 400 input features and 10 output classes

ypred = [fnn(x) for x in yy]
print(ypred[0].data) #(10,) shape for each image after passing through the fully connected layer





params = n.parameters() + fnn.parameters()

N_TRAIN    = 2000      # train on a subset first; all 50,000 will take a very long time
BATCH      = 32
LR         = 0.05      # can be larger than with batch size 1
EPOCHS     = 20

Xs, Ys = X_train[:N_TRAIN], y_train[:N_TRAIN]

for epoch in range(EPOCHS):
    order = np.random.permutation(N_TRAIN)        # shuffle each epoch
    total_loss, correct, t0 = 0.0, 0, time.time()

    for start in range(0, N_TRAIN, BATCH):
        idx = order[start:start + BATCH]

        for p in params:                          # zero once per batch
            p.grad = np.zeros_like(p.data)

        for i in idx:
            logits = fnn(n(Xs[i]).reshape(1, -1))
            loss = logits.cross_entropy_loss(Ys[i])
            loss.backward()                       # grads add up across the batch
            total_loss += loss.data.item()
            correct += int(np.argmax(logits.data) == Ys[i])

        for p in params:                          # one update with the average gradient
            p.data -= LR * p.grad / len(idx)

    print(f"epoch {epoch:2d}  loss {total_loss / N_TRAIN:.3f}  "
          f"acc {correct / N_TRAIN:.3f}  ({time.time() - t0:.0f}s)")




