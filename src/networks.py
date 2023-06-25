import torch
import torch.nn as nn

class DQN_2L60(nn.Module):

  name = "DQN_2L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_2L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.softmax(x)

    return x

class DQN_3L60(nn.Module):

  name = "DQN_3L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_3L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.softmax(x)

    return x

class DQN_3L100(nn.Module):

  name = "DQN_3L100"

  def __init__(self, inputs, outputs, device):
    super(DQN_3L100, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 100)
    self.linear2 = torch.nn.Linear(100, 100)
    self.linear3 = torch.nn.Linear(100, 100)
    self.linear4 = torch.nn.Linear(100, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.softmax(x)

    return x

class DQN_4L60(nn.Module):

  name = "DQN_4L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_4L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, 60)
    self.linear5 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.softmax(x)

    return x

class DQN_4L100(nn.Module):

  name = "DQN_4L100"

  def __init__(self, inputs, outputs, device):
    super(DQN_4L100, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 100)
    self.linear2 = torch.nn.Linear(100, 100)
    self.linear3 = torch.nn.Linear(100, 100)
    self.linear4 = torch.nn.Linear(100, 100)
    self.linear5 = torch.nn.Linear(100, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.softmax(x)

    return x

class DQN_5L60(nn.Module):

  name = "DQN_5L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_5L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, 60)
    self.linear5 = torch.nn.Linear(60, 60)
    self.linear6 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.activation(x)
    x = self.linear6(x)
    x = self.softmax(x)

    return x

class DQN_5L100(nn.Module):

  name = "DQN_5L100"

  def __init__(self, inputs, outputs, device):
    super(DQN_5L100, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 100)
    self.linear2 = torch.nn.Linear(100, 100)
    self.linear3 = torch.nn.Linear(100, 100)
    self.linear4 = torch.nn.Linear(100, 100)
    self.linear5 = torch.nn.Linear(100, 100)
    self.linear6 = torch.nn.Linear(100, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.activation(x)
    x = self.linear6(x)
    x = self.softmax(x)

    return x

class DQN_6L60(nn.Module):

  name = "DQN_6L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_6L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, 60)
    self.linear5 = torch.nn.Linear(60, 60)
    self.linear6 = torch.nn.Linear(60, 60)
    self.linear7 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.activation(x)
    x = self.linear6(x)
    x = self.activation(x)
    x = self.linear7(x)
    x = self.softmax(x)

    return x

class DQN_7L60(nn.Module):

  name = "DQN_7L60"

  def __init__(self, inputs, outputs, device):
    super(DQN_7L60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 60)
    self.linear2 = torch.nn.Linear(60, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, 60)
    self.linear5 = torch.nn.Linear(60, 60)
    self.linear6 = torch.nn.Linear(60, 60)
    self.linear7 = torch.nn.Linear(60, 60)
    self.linear8 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.activation(x)
    x = self.linear6(x)
    x = self.activation(x)
    x = self.linear7(x)
    x = self.activation(x)
    x = self.linear8(x)
    x = self.softmax(x)

    return x

class DQN_7L100(nn.Module):

  name = "DQN_7L100"

  def __init__(self, inputs, outputs, device):
    super(DQN_7L100, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 100)
    self.linear2 = torch.nn.Linear(100, 100)
    self.linear3 = torch.nn.Linear(100, 100)
    self.linear4 = torch.nn.Linear(100, 100)
    self.linear5 = torch.nn.Linear(100, 100)
    self.linear6 = torch.nn.Linear(100, 100)
    self.linear7 = torch.nn.Linear(100, 100)
    self.linear8 = torch.nn.Linear(100, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.activation(x)
    x = self.linear6(x)
    x = self.activation(x)
    x = self.linear7(x)
    x = self.activation(x)
    x = self.linear8(x)
    x = self.softmax(x)

    return x

# --- variable architecture --- #

class DQN_variable(nn.Module):

  name = "DQN_"

  def __init__(self, layers, device):

    super(DQN_variable, self).__init__()
    self.device = device

    self.linear = []
    for i in range(len(layers) - 1):
      self.linear.append(torch.nn.Linear(layers[i], layers[i + 1]))
      if i == 1: self.name += f"{layers[i]}"
      if i > 1: self.name += f"x{layers[i]}"

    self.linear = nn.ModuleList(self.linear)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):

    x = x.to(self.device)

    for i in range(len(self.linear) - 1):
      x = self.linear[i](x)
      x = self.activation(x)

    x = self.linear[i + 1](x)
    x = self.softmax(x)

    return x

# --- ideas --- #

class DQN_4L120_60(nn.Module):

  name = "DQN_4L120_60"

  def __init__(self, inputs, outputs, device):
    super(DQN_4L120_60, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, 120)
    self.linear2 = torch.nn.Linear(120, 60)
    self.linear3 = torch.nn.Linear(60, 60)
    self.linear4 = torch.nn.Linear(60, 60)
    self.linear5 = torch.nn.Linear(60, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.activation(x)
    x = self.linear5(x)
    x = self.softmax(x)

    return x

class DQN_2LX(nn.Module):

  name = "DQN_2L"

  def __init__(self, inputs, outputs, device, width):
    super(DQN_2LX, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, width)
    self.linear2 = torch.nn.Linear(width, width)
    self.linear3 = torch.nn.Linear(width, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)
    self.name += str(width)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.softmax(x)

    return x

class DQN_3LX(nn.Module):

  name = "DQN_3L"

  def __init__(self, inputs, outputs, device, width):
    super(DQN_3LX, self).__init__()
    self.device = device
    self.linear1 = torch.nn.Linear(inputs, width)
    self.linear2 = torch.nn.Linear(width, width)
    self.linear3 = torch.nn.Linear(width, width)
    self.linear4 = torch.nn.Linear(width, outputs)
    self.activation = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)
    self.name += str(width)

  def forward(self, x):
    x = x.to(self.device)
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    x = self.activation(x)
    x = self.linear3(x)
    x = self.activation(x)
    x = self.linear4(x)
    x = self.softmax(x)

    return x