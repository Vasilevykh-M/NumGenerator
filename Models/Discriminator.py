import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self, input_channel = 1, output_label = 10):
    super(Discriminator, self).__init__()
    self.input_channel = input_channel
    self.output_label = output_label
    self.relu = nn.LeakyReLU(0.2)
    self.dropout = nn.Dropout(0.3)
    self.linear1 = nn.Linear(784, 512)
    self.linear2 = nn.Linear(512, 256)
    self.linear3 = nn.Linear(256, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input_image):
    input_image = input_image.view(-1, 784)
    output = self.dropout(self.relu(self.linear1(input_image)))
    output = self.dropout(self.relu(self.linear2(output)))
    output = self.linear3(output)
    return self.sigmoid(output)