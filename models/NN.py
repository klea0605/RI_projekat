import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Sequential(
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 4)
    )
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.fc(x)
    preds = self.softmax(x)

    return preds
  
# --- Koristi se u kombinaciji sa csp transform --- #