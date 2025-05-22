# VAE Emotion Detection

This project implements a Variational Autoencoder (VAE) with emotion classification (EmotionVAE) for detecting emotions from speech audio using the RAVDESS dataset. The model processes MFCC features to reconstruct audio and predict one of eight emotions: neutral, calm, happy, sad, angry, fearful, disgust, or surprised.

### Features



* VAE-based audio feature learning.

* Supervised emotion classification.

* Training from scratch or resuming from checkpoints.

* PyTorch-based, CPU/GPU compatible.

### Installation

#### 1.Clone the Repository:

`git clone https://github.com/PhongNguyen-RhineLab/VAE-Emotion-Detection.git`

`cd vae-emotion-detection`

#### 2.Set Up Conda Environment:

* Install Anaconda or Miniconda.
* Create environment:

`conda env create -f environment.yml`

`conda activate VAE-Emotion-Detection`

### Usage

#### To start training from scratch use:

#### For RAVDESS:

`python VAE_train_extra.py --dataset ravdess --data_dir ./RAVDESS/audio_speech_actors_01-24 --epochs 50 --batch_size 32`

~~`python VAE_train_extra.py --data_dir ./archive/audio_speech_actors_01-24 --epochs 50 --batch_size 32`~~

Colab

`!python /content/VAE-Emotion-Detection/Colab-Support/VAE_train_extra.py --data_dir /content/VAE-Emotion-Detection/RAVDESS/audio_speech_actors_01-24 --epochs 50 --batch_size 32`

* Saves checkpoints to ./Checkpoint/vae_epoch_N.pth.

* Prints metrics per epoch.

#### For CREMA-D:

`python VAE_train_extra.py --dataset cremad --data_dir ./CREMA-D --epochs 50 --batch_size 32`

Colab

`!python /content/VAE-Emotion-Detection/Colab-Support/VAE_train.py --dataset cremad --data_dir ./CREMA-D --epochs 50 --batch_size 32`

#### To continue training a checkpoint use

`python VAE_train_extra.py --data_dir ./archive/audio_speech_actors_01-24 --epochs 50 --checkpoint ./epoch/vae_epoch_3.pth --start_epoch 2`

#### For Colab

`!python /content/VAE-Emotion-Detection/Colab-Support/VAE_train_extra.py --data_dir /content/VAE-Emotion-Detection/archive/audio_speech_actors_01-24 --epochs 50 --checkpoint /content/VAE-Emotion-Detection/epoch/vae_epoch_3.pth --start_epoch 2`

### Project Structure

* VAE_train_extra.py: Training script.
* Preprocess_RAVDESS.py: Dataset preprocessing for RAVDESS.
* Preprocess_CREMAD.py: Dataset preprocessing for CREMA-D.
* VAE_emotion_recognition.py: Model and loss function.
* Inference.py: Inference script.
* environment.yml: Conda environment configuration.
* Checkpoint/: Checkpoint directory.

### Requirements

* Conda with Python 3.11.11
* Hardware: CPU or CUDA-compatible GPU
* Dataset: RAVDESS (~1440 audio files), CREMA-D (~7,442 files)

### Colab Link:

[Colab](https://colab.research.google.com/drive/1BJ7kHSrqiIF6kz5iYQhZ2QvYJKJnW_Hl?usp=sharing)