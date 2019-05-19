import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    def __init__(self, input_dim = 784, hidden_dim=500, z_dim=20):
        super().__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.mean_layer  = nn.Linear(hidden_dim, z_dim)
        self.std_layer = nn.Linear(hidden_dim, z_dim)


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        hidden_out = self.hidden_layer(input)
        mean = self.mean_layer(hidden_out)
        std = torch.sqrt(torch.exp(self.std_layer(hidden_out)))

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, out_dim = 784):
        super().__init__()
        self.first_linear = nn.Sequential(
            nn.Linear(z_dim,hidden_dim),
            nn.Tanh()
        )

        self.second_linear = nn.Sequential(
            nn.Linear(hidden_dim,out_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.second_linear(self.first_linear(input))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, z_dim=z_dim)
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        batch_size, _ = input.shape
        mu, std = self.encoder.forward(input)
        eps = torch.randn((batch_size, self.z_dim), device=device)
        z = mu + std * eps
        output = self.decoder(z)

        loss_recon = torch.sum(self.loss(output, input)) / batch_size
        loss_reg = torch.sum(std ** 2 - torch.log(std ** 2) - 1 + mu ** 2) / (2*batch_size)

        average_negative_elbo = loss_recon + loss_reg

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims = self.decoder(torch.randn((n_samples, self.z_dim)).to(device))
        im_means = sampled_ims.mean(dim=0)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0
    for i, datapoints in enumerate(data):
        datapoints = datapoints.reshape(-1, 28 ** 2).to(device)

        elbo = model(datapoints)

        if model.training:
            model.zero_grad()
            elbo.backward()
            optimizer.step()


        average_epoch_elbo += elbo.item()

    average_epoch_elbo /= i

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        sampled_imgs, im_means = model.sample(25)
        sampled_imgs = sampled_imgs.detach()
        im_means = im_means.detach()
        sampled_imgs = sampled_imgs.reshape(25, 28, 28).to(device)
        save_dir = "images/vae/{}".format(epoch)
        gen_imgs_colour = torch.empty(25, 3, 28, 28)
        for i in range(3):
            gen_imgs_colour[:, i, :, :] = sampled_imgs
        grid = make_grid(gen_imgs_colour, nrow=5, normalize=True).permute(1, 2, 0)
        plt.imsave(save_dir, grid)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'images/vae/elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
