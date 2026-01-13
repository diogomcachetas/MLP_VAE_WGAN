'''
In a Wasserstein GAN (WGAN) setup, the discriminator loss being negative is actually expected and desired. 
In WGANs, the discriminator loss is not a traditional classification loss as in typical GANs. 
Instead, it's formulated as a Wasserstein distance or Earth Mover's Distance (EMD) between the distributions of real and generated samples.
The WGAN discriminator loss is defined as the difference between the mean output of the discriminator on real samples and the mean output on fake samples.
The goal is to minimize this difference, which effectively means maximizing the discriminator's ability to differentiate between real and fake samples while maintaining a Lipschitz constraint.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.optimizer import Optimizer
import numpy as np

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, num_classes=5):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            )

        # Latent Space: Mean and Log Variance Layers
        self.mean_layer = nn.Linear(64, latent_dim) 
        self.logvar_layer = nn.Linear(64, latent_dim) 

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 128),         
            nn.LeakyReLU(0.1),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),

            nn.Linear(512, input_dim),
            nn.LeakyReLU(0.1),

            nn.SELU()
        )

        # Classification branch MLP
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.01),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.01),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),

            nn.Linear(256, num_classes)
        )

    def reparameterize(self, mean, logvar):
        '''
        In a VAE, the encoder produces two outputs: the mean and the log-variance of the latent variables' distribution. 
        This is because VAEs assume that the latent variables follow a Gaussian distribution. 
        The challenge is to sample from this distribution in a way that allows gradients to propagate back through the network during training.
        By reparameterizing the sampling step in this way, we convert the stochastic sampling into a deterministic operation. 
        This allows the gradients to flow through during backpropagation, making the VAE trainable using standard optimization techniques.
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mean = self.mean_layer(encoded)
        logvar = self.logvar_layer(encoded)
        z = self.reparameterize(mean, logvar)
        decoded = self.decoder(z)
        
        #MLP
        classification = self.classifier(decoded)

        return decoded, mean, logvar, z, classification

# Discriminator architecture must be Lipschitz continuous DO NOT ADD ACTIVATION LAYER TO THE END
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),

            nn.Linear(512, 64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.model(x)

def vae_loss(x, x_decoded_mean, z_mean, z_log_var, classification, labels, epoch, num_epochs):
    import math
    classification_weight = 1.0 / (1.0 + math.exp(-0.005 * (epoch - (num_epochs * 1.25) / 2))) # 0.75/2 0.9/2 #750 # the lower is k, the lower the slope is # 0.5
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    mse_loss = F.mse_loss(x_decoded_mean, x, reduction='sum')
    mae_loss = F.l1_loss(x_decoded_mean, x, reduction='sum')
    classification_loss = F.cross_entropy(classification, labels, reduction='sum')
    #probs = F.softmax(classification, dim=1)
    #labels_one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    #classification_loss = torch.abs(probs - labels_one_hot).sum()
    full_loss =  (kl_loss + mae_loss * 5.0 + mse_loss) + (classification_loss * classification_weight)
    return full_loss, kl_loss, mae_loss, mse_loss, classification_loss

# Define Wassertein GAN loss
def wasserstein_loss(discriminator_output_real, discriminator_output_fake):
    '''
    Wasserstein distance, also known as the Earth Mover’s Distance (EMD), measures the minimum cost of transporting mass to transform one probability distribution into another.
    Provides more stable training and better convergence compared to traditional GANs.
    '''
    return -torch.sum(discriminator_output_real) + torch.sum(discriminator_output_fake)
    #return torch.sum(discriminator_output_real) - torch.sum(discriminator_output_fake)
    #return torch.abs(torch.sum(discriminator_output_real) - torch.sum(discriminator_output_fake))

def wasserstein_loss_g(discriminator_output_fake):
    return -torch.sum(discriminator_output_fake)

# Define gradient penalty function
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    '''
    The concept of gradient penalty is closely related to the Lipschitz continuity, which is a property ensuring that the discriminator doesn't change too rapidly. 
    Without this, the Wasserstein distance between the real and generated data distributions can become undefined or difficult to optimize.
    Which in turn facilitates stable and efficient training of the GAN model, leading to higher-quality generated samples.
    '''
    alpha = torch.rand(real_samples.size(0), *[1]*(real_samples.dim()-1), device=real_samples.device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad = True
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interpolates), 
                                    create_graph=True, only_inputs=True, retain_graph=True)[0] 
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 
    return gradient_penalty

 
def weights_init(m, s):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            torch.manual_seed(s)
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

    elif isinstance(m, nn.Conv1d):
        with torch.no_grad():
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(s)
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

def preprocess_data(df):
    features = df.iloc[:, :-3].values
    labels = df.iloc[:, -3].map({'control': 0, 'mixture': 1, 'sulfamethoxazole': 2, 'sulfapyridine': 3, 'sulfathiazole': 4}).values
    wavenumbers = df.columns[:-3].astype(float)
    concentration = None #df.iloc[:, -2].values
    return features, labels, wavenumbers, concentration

