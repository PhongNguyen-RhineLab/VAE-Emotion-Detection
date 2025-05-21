import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, n_mfcc=13, max_frames=100, split='train'):
        self.data_dir = data_dir
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames
        self.split = split

        # Map RAVDESS emotion codes to labels (0-based)
        self.emotion_map = {
            '01': 0,  # Neutral
            '02': 1,  # Calm
            '03': 2,  # Happy
            '04': 3,  # Sad
            '05': 4,  # Angry
            '06': 5,  # Fearful
            '07': 6,  # Disgust
            '08': 7  # Surprised
        }

        # Collect all audio files
        self.audio_files = []
        self.labels = []
        for actor_dir in os.listdir(data_dir):
            actor_path = os.path.join(data_dir, actor_dir)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.wav'):
                        # Parse filename (e.g., 03-01-01-01-01-01-01.wav)
                        parts = file.split('-')
                        if len(parts) >= 3:
                            emotion_code = parts[2]
                            if emotion_code in self.emotion_map:
                                self.audio_files.append(os.path.join(actor_path, file))
                                self.labels.append(self.emotion_map[emotion_code])

        # Split into train and validation sets
        train_files, val_files, train_labels, val_labels = train_test_split(
            self.audio_files, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )

        if split == 'train':
            self.audio_files = train_files
            self.labels = train_labels
        elif split == 'val':
            self.audio_files = val_files
            self.labels = val_labels
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load audio and extract MFCC
        y, sr = librosa.load(audio_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=2048, hop_length=512)

        # Normalize MFCC
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

        # Pad or truncate to max_frames
        if mfcc.shape[1] > self.max_frames:
            mfcc = mfcc[:, :self.max_frames]
        elif mfcc.shape[1] < self.max_frames:
            pad_width = self.max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

        # Convert to tensor and add channel dimension
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Shape: (1, n_mfcc, max_frames)

        return mfcc, label


if __name__ == "__main__":
    # Test the dataset
    dataset = RAVDESSDataset(data_dir="./archive/audio_speech_actors_01-24", n_mfcc=13, max_frames=100, split='train')
    mfcc, label = dataset[0]
    print(f"MFCC shape: {mfcc.shape}")  # Should print: torch.Size([1, 13, 100])
    print(f"Label: {label}")