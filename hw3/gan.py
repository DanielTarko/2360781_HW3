import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        channels = [in_size[0], 64, 128, 256]  # Channel sizes for conv layers
        
        # Build CNN layers
        cnn_layers = []
        for i in range(len(channels)-1):
            cnn_layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2)
            ])
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate number of features after CNN
        self.num_features = self._calc_num_cnn_features(in_size)
        
        # Final fully connected layer
        self.fc = nn.Linear(self.num_features, 1)
        # ========================

    def _calc_num_cnn_features(self, in_shape):
        with torch.no_grad():
            x = torch.zeros(1, *in_shape)
            out_shape = self.cnn(x).shape
        return int(np.prod(out_shape))

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        # CNN feature extraction
        features = self.cnn(x)
        # Flatten and pass through FC layer
        features = features.view(-1, self.num_features)
        # Keep the second dimension by removing squeeze
        y = self.fc(features)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim


        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        # Calculate initial feature map dimensions
        self.featuremap_size = featuremap_size
        initial_fm_size = featuremap_size * featuremap_size * 512
        
        # Project and reshape from z_dim to initial feature map
        self.fc = nn.Linear(z_dim, initial_fm_size)
        
        # Transposed convolution layers
        self.conv_layers = nn.Sequential(
            # Layer 1: 512 -> 256 channels
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 2: 256 -> 128 channels
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 3: 128 -> 64 channels
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final layer: 64 -> out_channels
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Map to [-1, 1] range
        )
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            z = torch.randn(n, self.z_dim, device=device)
            samples = self.forward(z)
            
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        x = self.fc(z)
        x = x.view(-1, 512, self.featuremap_size, self.featuremap_size)
        
        # Apply transposed convolutions
        x = self.conv_layers(x)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = y_data.device
    
    # Create labels with noise
    if data_label == 1:
        # Real data should be classified as 1
        noisy_data_labels = torch.ones(y_data.shape, device=device, dtype=torch.float)
        noisy_generated_labels = torch.zeros(y_data.shape, device=device, dtype=torch.float)
    else:
        # Real data should be classified as 0
        noisy_data_labels = torch.zeros(y_data.shape, device=device, dtype=torch.float)
        noisy_generated_labels = torch.ones(y_data.shape, device=device, dtype=torch.float)
    
    # Add uniform noise to labels if specified
    if label_noise > 0:
        noisy_data_labels += torch.rand(y_data.shape, device=device) * label_noise - label_noise/2
        noisy_generated_labels += torch.rand(y_data.shape, device=device) * label_noise - label_noise/2

    # Compute losses using BCE with logits
    criterion = nn.BCEWithLogitsLoss()
    loss_data = criterion(y_data, noisy_data_labels)
    loss_generated = criterion(y_generated, noisy_generated_labels)
    
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    device = y_generated.device
    
    # Generator wants discriminator to classify its samples as real
    target_labels = torch.full((y_generated.shape), 1-data_label if data_label == 0 else data_label, 
                             device=device, dtype=torch.float)
    
    # Compute loss using BCE with logits
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(y_generated, target_labels)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """
    dsc_model.train()
    gen_model.train()
    
    batch_size = x_data.shape[0]
    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    
    # Forward pass with real data
    dsc_real = dsc_model(x_data)
    
    # Generate fake data
    fake_data = gen_model.sample(batch_size, with_grad=True)
    
    # Forward pass with generated data
    dsc_generated = dsc_model(fake_data.detach())  # Detach to avoid generator gradient
    
    # Calculate discriminator loss
    dsc_loss = dsc_loss_fn(dsc_real, dsc_generated)
    
    # Update discriminator parameters
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    
    # Forward pass with new generated data through discriminator
    fake_data = gen_model.sample(batch_size, with_grad=True)
    dsc_generated = dsc_model(fake_data)  # Don't detach here
    
    # Calculate generator loss
    gen_loss = gen_loss_fn(dsc_generated)
    
    # Update generator parameters
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(gen_losses) < 2 or len(dsc_losses) < 2:    # ========================
        torch.save(gen_model, checkpoint_file)
        print(f"*** Saved checkpoint {checkpoint_file} ")
        saved = True
    elif (gen_losses[-1] < min(gen_losses[:-1]) or dsc_losses[-1] < min(dsc_losses[:-1])):    # ========================
        torch.save(gen_model, checkpoint_file)
        print(f"*** Saved checkpoint {checkpoint_file} ")
        saved = True
    return saved
