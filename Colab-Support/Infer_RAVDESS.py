import torch
import librosa
from VAE_emotion_recognition import EmotionVAE


def predict_emotion(model, audio_path, device, n_mfcc=13, max_frames=100):
    model.eval()
    emotion_map = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
                   4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}

    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=48000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_frames]
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        device)  # Shape: (1, 1, n_mfcc, max_frames)

    # Predict
    with torch.no_grad():
        _, mu, logvar, emotion_logits = model(mfcc)
        probabilities = torch.softmax(emotion_logits, dim=1)
        predicted = torch.argmax(probabilities, dim=1).item()

    return emotion_map[predicted], probabilities[0].cpu().numpy()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionVAE(input_shape=(13, 100), latent_dim=32, num_emotions=8).to(device)
    model.load_state_dict(torch.load("vae_epoch_50.pth"))

    audio_path = "path/to/test_audio.wav"
    emotion, probs = predict_emotion(model, audio_path, device)
    print(f"Predicted emotion: {emotion}")
    print(f"Probabilities: {probs}")