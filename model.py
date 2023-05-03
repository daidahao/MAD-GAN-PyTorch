import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, depth, activation):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(n_in, n_hidden, depth, batch_first=True)
        self.linear = nn.Linear(n_hidden, n_out)
        self.activation = activation()

class Discriminator(LSTM):
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        x = self.activation(x)
        return x

class Generator(LSTM):
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class MADGAN(nn.Module):
    def __init__(self, n_feats, n_window, n_hidden=100, n_latent=15):
        super(MADGAN, self).__init__()
        self.name = 'MAD-GAN'
        # hyper-parameters
        # n_feats: number of features
        self.n_feats = n_feats
        # n_window: time window size
        self.n_window = n_window
        # n_hidden: number of hidden units
        self.n_hidden = n_hidden
        # n_latent: number of latent features
        self.n_latent = n_latent
        # lstm as generator
        self.generator = Generator(self.n_latent, self.n_hidden, self.n_feats, 3, nn.Tanh)
        # lstm as discriminator
        self.discriminator = Discriminator(self.n_feats, self.n_hidden, 1, 1, nn.Sigmoid)
