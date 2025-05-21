# VAE-Emotion-Detection

To start training use:

`python VAE_train_extra.py --data_dir ./archive/audio_speech_actors_01-24 --epochs 50 --batch_size 32`

To continue training a checkpoint use

`python VAE_train_extra.py --data_dir ./archive/audio_speech_actors_01-24 --epochs 50 --checkpoint ./epoch/vae_epoch_3.pth --start_epoch 2`