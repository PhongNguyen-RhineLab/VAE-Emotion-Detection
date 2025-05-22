import torch
import librosa
import numpy as np
from VAE_emotion_recognition import EmotionVAE


def predict_emotion(model, audio_path, device, dataset='ravdess', n_mfcc=13, max_frames=100):
    model.eval()
    if dataset == 'ravdess':
        emotion_map = {
            0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
            4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
        }
    else:  # cremad
        emotion_map = {
            0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry',
            4: 'fearful', 5: 'disgust'
        }

    # Load and preprocess audio
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    if mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant')
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        _, _, _, emotion_logits = model(mfcc)
        probs = torch.softmax(emotion_logits, dim=1).squeeze().cpu().numpy()
        emotion_idx = torch.argmax(emotion_logits, dim=1).item()

    return emotion_map[emotion_idx], probs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['ravdess', 'cremad'], default='ravdess')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_emotions = 8 if args.dataset == 'ravdess' else 6
    model = EmotionVAE(input_shape=(13, 100), latent_dim=32, num_emotions=num_emotions).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    emotion, probs = predict_emotion(model, args.audio_path, device, dataset=args.dataset)
    print(f"Predicted emotion: {emotion}")
    print(f"Probabilities: {probs}")