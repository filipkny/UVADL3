import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Sigmoid() # choose a different output non linearity?
        )

    def forward(self, z):
        # Generate images from z
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # use different non linearity?
        )

    def forward(self, img):
        return self.model(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    discriminator = discriminator.to(device)
    generator = generator.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(args.n_epochs):
        print("Epoch {}".format(epoch))
        for i, (imgs, _) in enumerate(dataloader):
            real_labels = torch.ones((imgs.shape[0],1)).to(device)
            fake_labels = torch.zeros((imgs.shape[0],1)).to(device)

            imgs.cuda()
            imgs.to(device)

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            real_imgs = imgs.view(-1, 784).to(device)
            z = torch.randn(args.batch_size, args.latent_dim, device=device)
            real_predictions = discriminator.forward(real_imgs)
            fake_predictions = discriminator.forward(generator(z))

            # Compute loss
            loss_real = criterion(real_predictions, real_labels)
            loss_fake = criterion(fake_predictions, fake_labels)
            loss_d = loss_real + loss_fake

            optimizer_D.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizer_D.step()


            # Train Generator
            # ---------------
            z = torch.randn(args.batch_size, args.latent_dim,device=device)
            fake_imgs = generator(z).to(device)
            d_fake = discriminator(fake_imgs).to(device)
            loss_gen = criterion(d_fake, real_labels)

            optimizer_G.zero_grad()
            loss_gen.backward(retain_graph=True)
            optimizer_G.step()


            if i % 100 == 0:
                print("{}: loss discrimnator -> {}, loss generator -> {}, acc {}".format(i,loss_d,loss_gen,1))
            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                pass


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ),
                                                (0.5, ))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
