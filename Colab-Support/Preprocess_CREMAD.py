import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CREMADDataset(Dataset):
    def __init__(self, data_dir, split='train', n_mfcc=13, max_frames=100, test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.split = split
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames
        self.emotion_map = {
            'NEU': 0,  # Neutral
            'HAP': 1,  # Happy
            'SAD': 2,  # Sad
            'ANG': 3,  # Angry
            'FEA': 4,  # Fearful
            'DIS': 5  # Disgust
        }

        # Collect audio files and labels
        self.audio_files = []
        self.labels = []
        for file in os.listdir(data_dir):
            if file.endswith('.wav'):
                parts = file.split('_')
                if len(parts) >= 3 and parts[2] in self.emotion_map:
                    self.audio_files.append(os.path.join(data_dir, file))
                    self.labels.append(self.emotion_map[parts[2]])

        # Train/validation split
        train_files, val_files, train_labels, val_labels = train_test_split(
            self.audio_files, self.labels, test_size=test_size, random_state=random_state, stratify=self.labels
        )

        if split == 'train':
            self.audio_files = train_files
            self.labels = train_labels
        else:
            self.audio_files = val_files
            self.labels = val_labels

        print(f"{split.capitalize()} dataset size: {len(self.audio_files)}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        try:
            y, sr = librosa.load(audio_path, sr=22050)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.n_mfcc, self.max_frames), label

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        # Normalize MFCCs
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

        # Pad or truncate to max_frames
        if mfcc.shape[1] > self.max_frames:
            mfcc = mfcc[:, :self.max_frames]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.max_frames - mfcc.shape[1])), mode='constant')

        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Shape: (1, n_mfcc, max_frames)
        return mfcc, label


def get_data_loaders(data_dir, batch_size=32, n_mfcc=13, max_frames=100):
    train_dataset = CREMADDataset(data_dir, split='train', n_mfcc=n_mfcc, max_frames=max_frames)
    val_dataset = CREMADDataset(data_dir, split='val', n_mfcc=n_mfcc, max_frames=max_frames)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader