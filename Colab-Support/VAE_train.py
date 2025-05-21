import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Preprocess_RAVDESS import RAVDESSDataset
from VAE_emotion_recognition import EmotionVAE, vae_loss
import argparse


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


def get_data_loaders(data_dir, batch_size=32):
    train_dataset = RAVDESSDataset(data_dir, n_mfcc=13, max_frames=100, split='train')
    val_dataset = RAVDESSDataset(data_dir, n_mfcc=13, max_frames=100, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train Emotion VAE on RAVDESS')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to RAVDESS dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch (0-based)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionVAE(input_shape=(13, 100), latent_dim=32, num_emotions=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    beta = 1.0

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New checkpoint format
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
            start_epoch = checkpoint.get('epoch', args.start_epoch) + 1
            print(f"Resuming training from epoch {start_epoch} (new checkpoint format)")
        else:
            # Old checkpoint format (direct state_dict)
            model.load_state_dict(checkpoint)
            start_epoch = args.start_epoch + 1
            print(f"Resuming training from epoch {start_epoch} (old checkpoint format)")
    else:
        start_epoch = args.start_epoch

    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, beta)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train - Loss: {train_metrics[0]:.4f}, Recon: {train_metrics[1]:.4f}, "
              f"KL: {train_metrics[2]:.4f}, Class: {train_metrics[3]:.4f}, Acc: {train_metrics[4]:.2f}%")
        print(f"Val   - Loss: {val_metrics[0]:.4f}, Recon: {val_metrics[1]:.4f}, "
              f"KL: {val_metrics[2]:.4f}, Class: {val_metrics[3]:.4f}, Acc: {val_metrics[4]:.2f}%")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, f"/content/VAE-Emotion-Detection/epoch/vae_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    main()