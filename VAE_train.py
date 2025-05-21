import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Preprocess_RAVDESS import get_data_loaders
from VAE_emotion_recognition import EmotionVAE, vae_loss


def train_epoch(model, data_loader, optimizer, device, beta=1.0):
    model.train()
    total_loss, total_recon, total_kl, total_class = 0, 0, 0, 0
    correct, total = 0, 0

    for batch in data_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        recon, mu, logvar, emotion_logits = model(inputs)
        loss, recon_loss, kl_loss, class_loss = vae_loss(recon, inputs, mu, logvar, emotion_logits, labels, beta)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_class += class_loss.item()

        # Accuracy
        _, predicted = torch.max(emotion_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return (total_loss / len(data_loader.dataset),
            total_recon / len(data_loader.dataset),
            total_kl / len(data_loader.dataset),
            total_class / len(data_loader.dataset),
            100 * correct / total)


def evaluate(model, data_loader, device):
    model.eval()
    total_loss, total_recon, total_kl, total_class = 0, 0, 0, 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            recon, mu, logvar, emotion_logits = model(inputs)
            loss, recon_loss, kl_loss, class_loss = vae_loss(recon, inputs, mu, logvar, emotion_logits, labels)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_class += class_loss.item()

            _, predicted = torch.max(emotion_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (total_loss / len(data_loader.dataset),
            total_recon / len(data_loader.dataset),
            total_kl / len(data_loader.dataset),
            total_class / len(data_loader.dataset),
            100 * correct / total)


if __name__ == "__main__":
    # Setup
    data_dir = "./archive/audio_speech_actors_01-24"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionVAE(input_shape=(13, 100), latent_dim=32, num_emotions=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=32)
    epochs = 50
    beta = 1.0

    # Training loop
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, beta)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train - Loss: {train_metrics[0]:.4f}, Recon: {train_metrics[1]:.4f}, "
              f"KL: {train_metrics[2]:.4f}, Class: {train_metrics[3]:.4f}, Acc: {train_metrics[4]:.2f}%")
        print(f"Val   - Loss: {val_metrics[0]:.4f}, Recon: {val_metrics[1]:.4f}, "
              f"KL: {val_metrics[2]:.4f}, Class: {val_metrics[3]:.4f}, Acc: {val_metrics[4]:.2f}%")

        # Save model (optional)
        torch.save(model.state_dict(), f"./epoch/vae_epoch_{epoch + 1}.pth")