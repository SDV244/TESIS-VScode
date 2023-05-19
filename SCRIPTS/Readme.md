# Code Repository

This repository contains code related to audio data processing and training and testing various deep learning models for audio classification tasks. Below is a brief overview of the contents of each folder:

## 1. Data Preparation
- `dataset generation`: Code for creating the training and testing datasets. This code takes raw audio files and their corresponding labels and prepares them for further processing and model training.

## 2. Data Augmentation
- `time_masking`: Code for applying time masking to the audio data. Time masking involves randomly masking certain time intervals within an audio sample.
- `frequency_masking`: Code for applying frequency masking to the audio data. Frequency masking involves randomly masking certain frequency bands within an audio sample.

## 3. ETL (Extract, Transform, Load)
- `time_stretching`: Code for applying time stretching augmentation to the audio data. Time stretching involves modifying the speed of an audio sample without changing its pitch.

## 4. Pre-processing
- `spectrogram_preprocessing`: Code for pre-processing audio data and converting it into spectrograms. Spectrograms are visual representations of the frequencies present in an audio signal over time.

## 5. Model Training
- `train_densenet121`: Code for training the DenseNet121 model using the pre-processed spectrograms.
- `train_resnet50`: Code for training the ResNet50 model using the pre-processed spectrograms.
- `train_mobilenet`: Code for training the MobileNet model using the pre-processed spectrograms.
- `train_inception_v3`: Code for training the Inception V3 model using the pre-processed spectrograms.

## 6. Model Testing
- `test_densenet121`: Code for testing the performance of the trained DenseNet121 model on new audio samples.
- `test_resnet50`: Code for testing the performance of the trained ResNet50 model on new audio samples.
- `test_mobilenet`: Code for testing the performance of the trained MobileNet model on new audio samples.
- `test_inception_v3`: Code for testing the performance of the trained Inception V3 model on new audio samples.

Please refer to the individual code files for more detailed information on their usage and functionality.
