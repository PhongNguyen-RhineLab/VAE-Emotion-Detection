import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, n_mfcc=13, max_frames=100):
        self.data_dir = data_dir
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames
        self.emotion_map = {
            '01': 0, '02': 1, '03': 2, '04': 3,
            '05': 4, '06': 5, '07': 6, '08': 7
        }  # neutral, calm, happy, sad, angry, fearful, disgust, surprised
        self.files = []
        self.labels = []

        # Load files and labels
        for actor_dir in os.listdir(data_dir):
            actor_path = os.path.join(data_dir, actor_dir)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(actor_path, file)
                        emotion = file.split('-')[2]  # Extract emotion from filename
                        if emotion in self.emotion_map:
                            self.files.append(file_path)
                            self.labels.append(self.emotion_map[emotion])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        # Load audio
        audio, sr = librosa.load(file_path, sr=48000)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)

        # Pad or truncate to max_frames
        if mfcc.shape[1] < self.max_frames:
            pad_width = self.max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_frames]

        # Normalize MFCCs
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

        # Convert to tensor
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Shape: (1, n_mfcc, max_frames)
        label = torch.tensor(label, dtype=torch.long)

        return mfcc, label


def get_data_loaders(data_dir, batch_size=32, train_split=0.8):
    dataset = RAVDESSDataset(data_dir, n_mfcc=13, max_frames=100)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = "./archive/audio_speech_actors_01-24"
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=32)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")