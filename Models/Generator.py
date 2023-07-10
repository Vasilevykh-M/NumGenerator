import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, input_label = 10, output_size = 10):
    super(Generator, self).__init__()
    self.input_label = input_label
    self.output_size = output_size
    self.relu = nn.LeakyReLU(0.2)
    self.dropout = nn.Dropout(0.3)

    self.linear1 = nn.Linear(100, 128)
    self.linear2 = nn.Linear(128, 512)
    self.linear3 = nn.Linear(512, 784)
    self.tanh = nn.Tanh()

  def forward(self, input_label):
    output = self.relu(self.linear1(input_label))
    output = self.relu(self.linear2(self.dropout(output)))
    output = self.relu(self.linear3(self.dropout(output)))
    output = self.tanh(output)
    return output