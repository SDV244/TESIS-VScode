"""
Created on Sun Aug  7 14:50:41 2022

@author: Michael || Sebastian

Time Stretching augmentation script

"""
#%%
import librosa
import numpy as np
import os
import re
import soundfile as sf

input_dir = './AUDIO_INTC41/INCT41/'
annotation_dir = './ANNOTATIONS_INTC41/INCT41/'
output_dir = './new_Stretched/'
#%%
def time_stretch(file_path, stretch_factor):
    """
    Apply time stretching to an audio file.

    This function reads an audio file located at the specified file path and applies time stretching to the audio signal. Time stretching modifies the duration of the audio without changing the pitch. The resulting time-stretched audio signal is returned.

    Parameters
    ----------
    file_path (str): Path to the audio file.
    stretch_factor (float): Stretch factor to be applied to the audio. A stretch factor greater than 1 increases the duration, while a stretch factor less than 1 decreases the duration.

    Returns
    -------
    numpy.ndarray: Time-stretched audio signal.

    """
    y, sr = librosa.load(file_path, sr=None)
    y_stretch = librosa.effects.time_stretch(y, stretch_factor)
    return y_stretch
#%%
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            stretched_audio = time_stretch(file_path, 0.5) # Stretch factor of 1.5
            output_file_path = os.path.join(output_dir, file)
            sr = 24000
            sf.write(output_file_path, stretched_audio, sr)
#%%
for root, dirs, files in os.walk(annotation_dir):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                annotations = f.readlines()
            stretched_annotations = []
            for annotation in annotations:
                start_time, end_time, species = re.split('\s+', annotation.strip())
                stretched_start_time = str(float(start_time) * 1.816)
                stretched_end_time = str(float(end_time) * 1.816)
                stretched_annotation = '\t'.join([stretched_start_time, stretched_end_time, species])
                stretched_annotations.append(stretched_annotation)
            output_file_path = os.path.join(output_dir, 'Annotations_Stretched', file)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as f:
                f.write('\n'.join(stretched_annotations))



# %%
