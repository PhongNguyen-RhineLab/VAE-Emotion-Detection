import argparse
import torch
import torch.optim as optim
import numpy as np
from VAE_emotion_recognition import EmotionVAE, vae_loss
from Preprocess_RAVDESS import get_data_loaders as get_ravdess_loaders
from Preprocess_CREMAD import get_data_loaders as get_cremad_loaders


def train_epoch(model, data_loader, optimizer, device, beta=1.0):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_class = 0
    correct = 0
    total = 0

    for batch in data_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar, emotion_logits = model(inputs)
        loss, recon_loss, kl_loss, class_loss = vae_loss(recon_batch, inputs, mu, logvar, emotion_logits, labels, beta)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_recon += recon_loss.item() * inputs.size(0)
        total_kl += kl_loss.item() * inputs.size(0)
        total_class += class_loss.item() * inputs.size(0)

        _, predicted = torch.max(emotion_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    n = sum(len(batch[0]) for batch in data_loader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'kl': total_kl / n,
        'class': total_class / n,
        'acc': 100 * correct / total
    }


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_class = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            recon_batch, mu, logvar, emotion_logits = model(inputs)
            loss, recon_loss, kl_loss, class_loss = vae_loss(recon_batch, inputs, mu, logvar, emotion_logits, labels)

            total_loss += loss.item() * inputs.size(0)
            total_recon += recon_loss.item() * inputs.size(0)
            total_kl += kl_loss.item() * inputs.size(0)
            total_class += class_loss.item() * inputs.size(0)

            _, predicted = torch.max(emotion_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    n = sum(len(batch[0]) for batch in data_loader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'kl': total_kl / n,
        'class': total_class / n,
        'acc': 100 * correct / total
    }


def main():
    parser = argparse.ArgumentParser(description='Train Emotion VAE')
    parser.add_argument('--dataset', type=str, choices=['ravdess', 'cremad'], default='ravdess',
                        help='Dataset to use (ravdess or cremad)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting Checkpoint (0-based)')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    import random
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select dataset and number of emotions
    if args.dataset == 'ravdess':
        train_loader, val_loader = get_ravdess_loaders(args.data_dir, args.batch_size)
        num_emotions = 8
    else:  # cremad
        train_loader, val_loader = get_cremad_loaders(args.data_dir, args.batch_size)
        num_emotions = 6

    # Initialize model
    model = EmotionVAE(input_shape=(13, 100), latent_dim=32, num_emotions=num_emotions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resumed from checkpoint: {args.checkpoint}")

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, beta=1.0)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon']:.4f}, "
              f"KL: {train_metrics['kl']:.4f}, Class: {train_metrics['class']:.4f}, Acc: {train_metrics['acc']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Recon: {val_metrics['recon']:.4f}, "
              f"KL: {val_metrics['kl']:.4f}, Class: {val_metrics['class']:.4f}, Acc: {val_metrics['acc']:.2f}%")

        # Save checkpoint
        torch.save({
            'Checkpoint': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, f"/content/VAE-Emotion-Detection/Checkpoint/vae_epoch_{epoch + 1}.pth")


if __name__ == '__main__':
    main()