import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

from Models import Discriminator, Generator
from Train import train

device = "cuda" if torch.cuda.is_available() else "cpu"
mnist_dataset = MNIST(root = 'data/', download = True, train = True,
                      transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]))

def generate_fake_images(model, size):
  matrix_lable = (torch.rand(128).repeat(size, 1)-0.5)/0.5
  return model(matrix_lable.to(device))

if __name__ == '__main__':

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mnist_dataset = MNIST(root='data', download=True,
                          train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]))

    batch_size = 128
    train_loader = DataLoader(mnist_dataset, batch_size, shuffle=True)

    epochs = 200
    lr = 0.0002

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    loss = nn.BCELoss()
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))


    train_loader = DataLoader(mnist_dataset, batch_size, shuffle=True)

    train(train_loader, epochs, lr, discriminator, generator, device)


