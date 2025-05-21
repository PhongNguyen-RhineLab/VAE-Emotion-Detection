import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionVAE(nn.Module):
    def __init__(self, input_shape=(13, 100), latent_dim=32, num_emotions=8):
        super(EmotionVAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_emotions = num_emotions

        # Encoder: Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)

        # Flat size calculation
        self.flat_size = 128 * ((input_shape[0] + 3) // 4) * ((input_shape[1] + 1) // 4)  # 128 * 4 * 25 = 12800

        # Encoder: Dense layers for mean and log-variance
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

        # Decoder: Dense layer to project latent space
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)

        # Decoder: Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=(0, 1))
        self.deconv3 = nn.Conv2d(32, 1, kernel_size=(3, 1), stride=1, padding=(0, 0))  # Adjusted to output (13, 100)

        # Emotion classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_emotions)
        )

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.flat_size)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_decode(z))
        x = x.view(-1, 128, (self.input_shape[0] + 3) // 4, (self.input_shape[1] + 1) // 4)
        x = F.relu(self.deconv1(x))  # (batch_size, 64, 8, 50)
        x = F.relu(self.deconv2(x))  # (batch_size, 32, 14, 100)
        x = self.deconv3(x)  # (batch_size, 1, 12, 100)
        # Crop to ensure exact output shape
        x = x[:, :, :13, :]  # Ensure height is 13
        return x

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        emotion_logits = self.classify(z)
        return reconstructed, mu, logvar, emotion_logits


# Loss function
def vae_loss(reconstructed, original, mu, logvar, emotion_logits, labels, beta=1.0):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Classification loss (Cross-Entropy)
    class_loss = F.cross_entropy(emotion_logits, labels, reduction='sum')

    # Total loss
    total_loss = recon_loss + beta * kl_loss + class_loss
    return total_loss, recon_loss, kl_loss, class_loss


# Example usage
def train(model, data_loader, optimizer, device, beta=1.0):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        recon, mu, logvar, emotion_logits = model(inputs)
        loss, recon_loss, kl_loss, class_loss = vae_loss(recon, inputs, mu, logvar, emotion_logits, labels, beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader.dataset)


if __name__ == "__main__":
    # Example model initialization
    model = EmotionVAE(input_shape=(13, 100), latent_dim=32, num_emotions=8)
    print(model)