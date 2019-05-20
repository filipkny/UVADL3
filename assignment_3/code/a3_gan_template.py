import argparse
import datetime
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE = True
sess_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

class Generator(nn.Module):
    def __init__(self, dropout = False):
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

        if dropout:
            self.model = nn.Sequential(
                nn.Linear(args.latent_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(1024, 784),
                nn.Tanh()  # choose a different output non linearity?
            )
        else:
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
                nn.Tanh() # choose a different output non linearity?
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
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid() # use different non linearity?
        )

    def forward(self, img):
        return self.model(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    print("device:", device)

    # Init models and loss functioon
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Data saving variables
    d_losses = []
    g_losses = []
    avg_d_losses = []
    avg_g_losses = []

    for epoch in range(args.n_epochs):
        print("Epoch {}".format(epoch))
        for i, (imgs, _) in enumerate(dataloader):

            imgs.cuda()
            imgs.to(device)

            # Train Generator
            # ---------------

            # Generate fake images from noise z
            z = torch.randn(args.batch_size, args.latent_dim,device=device)
            fake_imgs = generator(z).to(device)

            # Use descriminator to make predictions and then compute generator loss
            fake_predictions = discriminator(fake_imgs).to(device)
            log_probs_fake = torch.log(fake_predictions)
            loss_gen = - torch.mean(log_probs_fake)
            g_losses.append(loss_gen.data.item())

            # Train generator
            optimizer_G.zero_grad()
            loss_gen.backward(retain_graph=True)
            optimizer_G.step()

            # Train Discriminator
            # -------------------

            # Get predictions on real images and compute respective loss
            real_imgs = imgs.view(-1, 784).to(device)
            real_predictions = discriminator.forward(real_imgs)

            # Get predictions on fake images and compute respective loss
            z = torch.randn(args.batch_size, args.latent_dim, device=device)
            fake_images = generator(z)
            fake_predictions = discriminator.forward(fake_images)

            log_probs_real = torch.log(real_predictions)
            real_loss  = - torch.mean(log_probs_real)
            log_probs_fake = torch.log(1 - fake_predictions)
            fake_loss = - torch.mean(log_probs_fake)

            # Compute total loss
            loss_d = real_loss + fake_loss
            d_losses.append(loss_d.data.item())

            # Train disriminator
            optimizer_D.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizer_D.step()


            # Print progress
            if i % 100 == 0:
                print("{}: loss discrimnator -> {}, loss generator -> {}".format(i,loss_d,loss_gen))

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                print("--- Saving images ---")
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                z = torch.from_numpy(np.random.normal(0, 1, (25, 100))).to(device=device, dtype=torch.float)
                img_sample = generator.forward(z).view(25, 28, 28).detach()
                gen_imgs_colour = torch.empty(25, 3, 28, 28)
                for i in range(3):
                    gen_imgs_colour[:, i, :, :] = img_sample

                grid = make_grid(gen_imgs_colour, nrow=5, normalize=True).permute(1, 2, 0)
                plot_name = "images/gan/image_{}.png"
                plt.imsave(plot_name, grid)

        avg_d_losses.append(np.mean(d_losses))
        avg_g_losses.append(np.mean(g_losses))

    if SAVE:
        to_save = [d_losses, g_losses, avg_d_losses, avg_g_losses]
        with open('images/gan/gan_losses_{}'.format(str(sess_id)), 'wb') as fp:
            pickle.dump(to_save, fp)

        torch.save(generator.state_dict(), "images/gan/models/gen_{}".format(sess_id))
        torch.save(discriminator.state_dict(), "images/gan/models/dis_{}".format(sess_id))



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
