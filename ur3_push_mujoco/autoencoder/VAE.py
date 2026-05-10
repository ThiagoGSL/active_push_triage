import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self, latent_dim=6):
        super().__init__()
        self.kl_loss = 0

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.fc2_mean = nn.Linear(in_features=256, out_features=latent_dim)
        self.fc2_logvar = nn.Linear(in_features=256, out_features=latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        z_mean = self.fc2_mean(x)
        z_log_var = self.fc2_logvar(x) 
        # loss
        self.kl_loss = torch.mean(torch.sum(-0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)), axis=1))

        return z_mean

class Decoder(nn.Module):
    def __init__(self, latent_dim=6):
        super().__init__()
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1024)
        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=32, kernel_size=4, stride=2)
        self.convt2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.convt3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2)
        self.convt4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=6, stride=2)

    
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = nn.Unflatten(1,(256, 2, 2))(z)
        z = F.relu(self.convt1(z))
        z = F.relu(self.convt2(z))
        z = F.relu(self.convt3(z))
        x = torch.sigmoid(self.convt4(z)) # sigmoid: make sure that output is between 0 and 1

        return x
    
class VAE(nn.Module):
    def __init__(self, latent_dim=6):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.reconstruction_loss = nn.BCELoss(reduction='none')

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x
 
    def compute_loss(self, data):
        # forward pass
        z = self.encoder(data)
        # reconstruction loss
        reconstruction = self.decoder(z)
        rec_loss = torch.mean(torch.sum(self.reconstruction_loss(reconstruction, data), axis=(2,3)))
        # total_loss
        total_loss = rec_loss + self.encoder.kl_loss

        return total_loss, rec_loss, self.encoder.kl_loss

if __name__ == "__main__":
    # debug
    random_img = torch.rand((1, 1, 64, 64))

    encoder = Encoder()
    enc_res = encoder(random_img)
    print("encoder res: ", enc_res)
    print("shape: ", enc_res.shape)
    print("="*100)

    decoder = Decoder()
    dec_res = decoder(enc_res)
    print("decoder res: ", dec_res)
    print("shape: ", dec_res.shape)