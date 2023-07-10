import torch
from torch.autograd import Variable
import torch.nn as nn

from Vizualization import vizualization


def train_discriminator(x, discriminator, generator, optimizerD, loss, device):

  optimizerD.zero_grad()

  targets_real = Variable(torch.ones(x.size()[0], 1).type(torch.FloatTensor).to(device))
  pred_real = discriminator(x.view(-1, 784).to(device))
  D_loss_real = loss(pred_real, targets_real)

  targets_fake = Variable(torch.zeros(x.size()[0], 1).type(torch.FloatTensor).to(device))
  matrix_lable = Variable(torch.randn(100).repeat(x.size()[0], 1).to(device))
  fake_images = generator(matrix_lable)
  pred_fake = discriminator(fake_images)
  D_loss_fake = loss(pred_fake, targets_fake)

  D_loss = (D_loss_real + D_loss_fake) / 2

  D_loss.backward()
  optimizerD.step()

  return D_loss.data.item()


def train_generator(x, discriminator, generator, optimizerG, loss, device):

  optimizerG.zero_grad()

  targets = Variable(torch.ones(x.size()[0], 1).type(torch.FloatTensor).to(device))
  matrix_lable = Variable(torch.randn(100).repeat(x.size()[0], 1).to(device))
  fake_images = generator(matrix_lable)
  pred = discriminator(fake_images)
  G_loss = loss(pred, targets)

  G_loss.backward()
  optimizerG.step()

  return G_loss.data.item()


def train(train_loader, epochs, lr, discriminator, generator, device):

  sample_z_in_train = torch.randn(64, 100, dtype=torch.float32).to(device)
  discriminator.train()
  generator.train()

  loss = nn.BCELoss()
  optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
  optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

  for epoch in range(epochs):
    D_train_loss = 0
    G_train_loss = 0

    for X, _ in train_loader:
      D_train_loss += train_discriminator(X, discriminator, generator, optimizerD, loss, device)
      G_train_loss += train_generator(X, discriminator, generator, optimizerG, loss, device)

    sample_gen_imgs_in_train = generator(sample_z_in_train).detach().view(64, 1, 28, 28).cpu()

    with torch.no_grad():
      vizualization(epoch, sample_gen_imgs_in_train)
      print(
        "[Epoch: %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch + 1, epochs, D_train_loss / len(train_loader), G_train_loss / len(train_loader))
      )


  torch.save(discriminator.state_dict(), "discriminator.pth")
  torch.save(generator.state_dict(), "generator.pth")