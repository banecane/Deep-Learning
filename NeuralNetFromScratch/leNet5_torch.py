# LeNet-5 on CIFAR-10 in PyTorch: the same network as leNet5.py,
# with comments pointing at the matching piece of the from-scratch engine.

import os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = r"C:\Users\nilak\OneDrive\Desktop\Programming\python\datasets\cifar-10-python\cifar-10-batches-py"
torch.manual_seed(0)


# ---------------------------------------------------------------- data
def load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    # PyTorch layers expect (N, C, H, W), which is exactly how the pickle stores it,
    # so no transpose here (your engine used (H, W, C)).
    X = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return X, np.array(d[b"labels"])

parts = [load_batch(os.path.join(DATA_DIR, f"data_batch_{i}")) for i in range(1, 6)]
X_train = torch.tensor(np.concatenate([p[0] for p in parts]))       # (50000, 3, 32, 32) float32
y_train = torch.tensor(np.concatenate([p[1] for p in parts]))       # (50000,) int64
X_test, y_test = (torch.tensor(a) for a in load_batch(os.path.join(DATA_DIR, "test_batch")))

# DataLoader replaces your  order = np.random.permutation(...)  +  order[start:start+BATCH]
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test),   batch_size=500)


# ---------------------------------------------------------------- model
class LeNet5(nn.Module):                       # nn.Module ~ your c_mln / MLP classes
    def __init__(self):
        super().__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size) ~ c_layer: a whole set of c_neurons,
        # each filter's weights + bias, already initialised with a fan_in-based scale.
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)    # (3,32,32) -> (6,28,28)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # (6,14,14) -> (16,10,10)
        # nn.Linear(nin, nout) ~ MLP.layer: one weight matrix + bias
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):                      # forward ~ your __call__
        x = F.max_pool2d(torch.tanh(self.conv1(x)), 2)   # conv -> tanh -> maxPool(2, 2)
        x = F.max_pool2d(torch.tanh(self.conv2(x)), 2)   # -> (N, 16, 5, 5)
        x = x.flatten(1)                                 # ~ reshape(1, -1), but keeps the batch axis
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)                               # raw logits, no tanh (same fix as before)

model = LeNet5()
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))   # ~ n.parameters() + fnn.parameters()

loss_fn = nn.CrossEntropyLoss()      # ~ cross_entropy_loss: stable softmax + -log p, mean over the batch
opt = torch.optim.SGD(model.parameters(), lr=0.05)   # ~ your  p.data -= LR * p.grad  loop


def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():            # no graph building: like using .data only
        for xb, yb in loader:
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
    model.train()
    return correct / len(loader.dataset)


# ---------------------------------------------------------------- training
EPOCHS = 10
for epoch in range(EPOCHS):
    t0, total, correct = time.time(), 0.0, 0
    for xb, yb in train_loader:      # xb: (64, 3, 32, 32), the WHOLE batch in one forward pass
        logits = model(xb)           # (64, 10)
        loss = loss_fn(logits, yb)   # one scalar: mean loss over the batch

        opt.zero_grad()              # ~ p.grad = np.zeros_like(p.data)
        loss.backward()              # same idea as your backward(): topo sort + chain rule
        opt.step()                   # ~ p.data -= LR * p.grad

        total += loss.item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()

    n = len(train_loader.dataset)
    print(f"epoch {epoch:2d}  loss {total / n:.3f}  train acc {correct / n:.3f}  "
          f"test acc {evaluate(test_loader):.3f}  ({time.time() - t0:.0f}s)")
