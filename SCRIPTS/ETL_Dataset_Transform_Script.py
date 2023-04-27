# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 14:50:41 2022

@author: Michael || Sebastian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build dataset

Audio recordings come into monophonic recordings. Manual annotations come into 
time-frequency segmentation performed using audacity or raven. There is one annotation
file per audio recording.

This code is a Python script for processing audio files and generating spectrograms and annotations.
The script starts by importing necessary libraries including Numpy, Pandas, Librosa, Matplotlib, etc.
It then sets some global variables such as the target frequency, window length, paths to audio and annotations, etc.
The code includes several functions: filter_window filters a dataframe based on start, end, and step time values and returns a list
of dataframes and a dictionary of dataframes grouped by the file name. filter_label filters dataframe rows based on specific labels. 
save_fname_lists saves data in the fname_lists dictionary to CSV files.

"""
#%%
# Loading libraries
import numpy as np  
import pandas as pd
from maad import sound, util
import glob
from os import walk
import sphinx
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import wave

print ("Done loading libraries")

#%%
# Set main variables
target_fs = 24000  # target fs of project
wl = 5  # Window length for formated rois
path_annot = '../ANNOTATIONS_INTC41/INCT41/'  # location of bbox annotations
path_audio = '../AUDIO_INTC41/INCT41/'  # location of raw audio
path_save = '../train_dataset/'  # location to save new samples
nombre = next(walk(r"C:\Users\sebas\Documents\GitHub\TESIS-VScode\ANNOTATIONS_INTC41\INCT41"), (None, None, []))[2]  # [] if no file
df = pd.DataFrame()

# _______TRIMMING AUDIO SETTINGS _______
# Set the length of each audio chunk in seconds
chunk_length = 5
# Specify the source and destination directories for the audio files
src_dir = '../AUDIO_INTC41/INCT41/'
dst_dir = '../NEW_Orig_Data/AUDIO_TRIM'

#________PATHS FOR GETTING SPECTROGRAM FILES________
input_folder = '../NEW_Orig_Data/AUDIO_TRIM/'
output_folder = '../NEW_Orig_Data/SPECTROGRAMS_IMAGES/'


#_______SETTING UP THE SPECTROGRAM FOLDER TO GENERATE THE DATAFRAME WITH THE INFORMATION_______
spectrogram_folder = '../NEW_Orig_Data/SPECTROGRAMS_IMAGES/'

#_______SETTING UP THE LABEL DATAFRAME PATH TO GENERATE THE DATAFRAME WITH THE INFORMATION_______
input_folder_LBL = '../NEW_Orig_Data/'

#%%
#________________________BEGINNING OF FUNCTIONS_______________


# def filter_window(df, start, end, step):
#     """
#     Filter a dataframe to get windows of time periods.

#     This function filters a dataframe based on start, end and step time values, creating a new dataframe for each
#     time window. The resulting dataframes are grouped by the file name and stored in a dictionary.

#     Parameters
#     ----------
#     df (pandas.DataFrame): Dataframe to be filtered.
#     start (int): Start time value in seconds.
#     end (int): End time value in seconds.
#     step (int): Step value in seconds to create new time windows.

#     Returns
#     -------
#     tuple: A tuple containing:
#         df_mlabel (list): A list of dataframes containing only rows with time between start and end values.
#         fname_lists (dict): A dictionary of dataframes grouped by the file name.
#     """
#     df_mlabel = []
#     fname_lists = {}
#     for x in range(start, end, step):
#         df_windowed = df[(df['min_t'] >= x) & (df['max_t'] <= x + step)]
#         df_mlabel.append(df_windowed)
#         for fname, group in df_windowed.groupby('fname'):
#             if fname not in fname_lists:
#                 fname_lists[fname] = [group]
#             else:
#                 fname_lists[fname].append(group)
#     return df_mlabel, fname_lists

# NEW ATEMPT
def filter_window(df, start, end, step):
    """
    Filter a dataframe to get windows of time periods.

    This function filters a dataframe based on start, end and step time values, creating a new dataframe for each
    time window. The resulting dataframes are grouped by the file name and stored in a dictionary.

    Parameters
    ----------
    df (pandas.DataFrame): Dataframe to be filtered.
    start (int): Start time value in seconds.
    end (int): End time value in seconds.
    step (int): Step value in seconds to create new time windows.

    Returns
    -------
    tuple: A tuple containing:
        df_mlabel (list): A list of dataframes containing only rows with time between start and end values.
        fname_lists (dict): A dictionary of dataframes grouped by the file name.
    """
    df_mlabel = []
    fname_lists = {}
    for x in range(start, end, step):
        df_windowed = df[(df['min_t'] >= x) & (df['max_t'] <= x + step)]
        df_mlabel.append(df_windowed)
        for fname in df_windowed['fname'].unique():
            if fname not in fname_lists:
                fname_lists[fname] = [df_windowed.loc[df_windowed['fname'] == fname]]
            else:
                fname_lists[fname].append(df_windowed.loc[df_windowed['fname'] == fname])
        #Handle annotations longer than step seconds
        df_long = df[(df['min_t'] < x) & (df['max_t'] > x + step)]
        for i, row in df_long.iterrows():
            start_time = max(row['min_t'], x)
            end_time = min(row['max_t'], x + step)
            while end_time - start_time >= step:
                new_row = row.copy()
                new_row['min_t'] = start_time
                new_row['max_t'] = start_time + step
                df_mlabel.append(pd.DataFrame([new_row]))
                if row['fname'] not in fname_lists:
                    fname_lists[row['fname']] = [pd.DataFrame([new_row])]
                else:
                    fname_lists[row['fname']].append(pd.DataFrame([new_row]))
                start_time += step
            if end_time - start_time > 0:
                new_row = row.copy()
                new_row['min_t'] = start_time
                new_row['max_t'] = end_time
                df_mlabel.append(pd.DataFrame([new_row]))
                if row['fname'] not in fname_lists:
                    fname_lists[row['fname']] = [pd.DataFrame([new_row])]
                else:
                    fname_lists[row['fname']].append(pd.DataFrame([new_row]))
    return df_mlabel, fname_lists



# FINAL OF NEW ATEMPT
    
#ATTEMPT_AGGAN

# def filter_window(df, start, end, step):
#     """
#     Filter a dataframe to get windows of time periods.

#     This function filters a dataframe based on start, end and step time values, creating a new dataframe for each
#     time window. The resulting dataframes are grouped by the file name and stored in a dictionary.

#     Parameters
#     ----------
#     df (pandas.DataFrame): Dataframe to be filtered.
#     start (int): Start time value in seconds.
#     end (int): End time value in seconds.
#     step (int): Step value in seconds to create new time windows.

#     Returns
#     -------
#     tuple: A tuple containing:
#         df_mlabel (list): A list of dataframes containing only rows with time between start and end values.
#         fname_lists (dict): A dictionary of dataframes grouped by the file name.
#     """
#     df_mlabel = []
#     fname_lists = {}

#     # Create an empty dataframe for each file name
#     for fname in df['fname'].unique():
#         fname_lists[fname] = pd.DataFrame()

#     for x in range(start, end, step):
#         # Filter the dataframe to get rows within the current time window
#         df_windowed = df[(df['min_t'] >= x) & (df['max_t'] <= x + step)]

#         # Append the filtered rows to the list of dataframes
#         df_mlabel.append(df_windowed)

#         # Group the filtered rows by file name
#         for fname, group in df_windowed.groupby('fname'):
#             fname_lists[fname] = pd.concat([fname_lists[fname], group], axis=0)

#         # Handle annotations longer than step seconds
#         df_long = df[(df['min_t'] < x) & (df['max_t'] > x + step)]
#         for i, row in df_long.iterrows():
#             start_time = max(row['min_t'], x)
#             end_time = min(row['max_t'], x + step)
#             while end_time - start_time >= step:
#                 new_row = row.copy()
#                 new_row['min_t'] = start_time
#                 new_row['max_t'] = start_time + step
#                 df_mlabel.append(pd.DataFrame([new_row]))
#                 fname_lists[row['fname']].append(pd.DataFrame([new_row]), ignore_index=True)
#                 start_time += step
#             if end_time - start_time > 0:
#                 new_row = row.copy()
#                 new_row['min_t'] = start_time
#                 new_row['max_t'] = end_time
#                 df_mlabel.append(pd.DataFrame([new_row]))
#                 fname_lists[row['fname']].append(pd.DataFrame([new_row]), ignore_index=True)

#     return df_mlabel, fname_lists


#ATTEMPTFIN
def filter_label(row):
    """
       Filter the dataframe row based on specific labels.

    Parameters
    ----------
    row : pd.Series
        The row of the dataframe that needs to be filtered.

    Returns
    -------
    bool
        True if the label in the row is one of the specific labels ('PHYCUV_M', 'PHYCUV_F','BOAALB_M', 'BOAALB_F', 'BOALUN_F', 'BOALUN_M', 'PHYCUV_M', 'PHYCUV_F'), otherwise False.

    """
    return row['label'] in ['PHYCUV_M', 'PHYCUV_F','BOAALB_M', 'BOAALB_F', 'BOALUN_F', 'BOALUN_M', 'PHYCUV_M', 'PHYCUV_F']


  
# def save_fname_lists(fname_lists, save_path):
#     """
#         This function saves the data in the `fname_lists` dictionary to csv files. 

#     Parameters
#     ----------
#     fname_lists (dict): A dictionary where the key is a string representing the name of an audio file,
#                         and the value is a list of dataframes.
#     save_path (str): The path to the directory where the csv files should be saved.

#     Returns
#     -------
#     None

#     Example:
#     save_fname_lists(fname_lists, "data/csv_files")
#     """
#     for fname, dfs in fname_lists.items():
#         dir_path = os.path.join(save_path, fname)
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#         for i, df in enumerate(dfs):
#             #file_path = os.path.join(dir_path, f"{fname.split('.')[0]}_{i}.csv")
#             file_path = os.path.join(dir_path, f"{fname.split('.')[0]}_{i}.csv").replace('\\', '/')

#             df.to_csv(file_path, index=False) 
 
def save_fname_lists(fname_lists, save_path):
    """
        This function saves the data in the `fname_lists` dictionary to csv files. 

    Parameters
    ----------
    fname_lists (dict): A dictionary where the key is a string representing the name of an audio file,
                        and the value is a list of dataframes.
    save_path (str): The path to the directory where the csv files should be saved.

    Returns
    -------
    None

    Example:
    save_fname_lists(fname_lists, "data/csv_files")
    """
    for fname, dfs in fname_lists.items():
        dir_path = os.path.join(save_path, fname)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for i, df in enumerate(dfs):
            if i < len(dfs):
                #file_path = os.path.join(dir_path, f"{fname.split('.')[0]}_{i}.csv")
                file_path = os.path.join(dir_path, f"{fname.split('.')[0]}_{i}.csv").replace('\\', '/')
                df.to_csv(file_path, index=False)


def trim_audio_files(src_dir, dst_dir, chunk_length):
    """
        Trim the audio files in `src_dir` into smaller chunks of `chunk_length` seconds and save them in `dst_dir`.

    Each audio file in `src_dir` will be divided into chunks of `chunk_length` seconds and saved in a new folder with the same name as the original file in `dst_dir`. Each chunk will have a unique filename consisting of the original filename followed by an underscore and a chunk index.

    Parameters
    ----------
    src_dir : str
        The path to the source directory containing the audio files.
    dst_dir : str
        The path to the destination directory where the trimmed audio chunks will be saved.
    chunk_length : float
        The length of each audio chunk in seconds.

    Returns
    -------
    None

    Notes
    -----
    The function assumes that all audio files in `src_dir` are in WAV format.
    """
    # Loop through the audio files in the source directory
    for filename in os.listdir(src_dir):
        if filename.endswith(".wav"):
            # Open the audio file
            with wave.open(os.path.join(src_dir, filename), 'rb') as audio_file:
                # Get the number of frames in the audio file
                num_frames = audio_file.getnframes()
                # Get the frame rate of the audio file
                frame_rate = audio_file.getframerate()
                #print(frame_rate)
                # Calculate the number of chunks in the audio file
                num_chunks = num_frames // (frame_rate * chunk_length)
                #print(num_chunks)
                # Make a folder for each audio file
                file_dir = os.path.join(dst_dir, filename.split(".")[0])
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)

                # Loop through the chunks of audio
                for chunk_index in range(num_chunks + 1):
                    # Create a new audio file for each chunk
                    chunk_filename = f"{filename.split('.')[0]}_{chunk_index}.wav"
                    chunk_file_path = os.path.join(file_dir, chunk_filename)
                    with wave.open(chunk_file_path, 'wb') as chunk_file:
                        chunk_file.setnchannels(audio_file.getnchannels())
                        chunk_file.setsampwidth(audio_file.getsampwidth())
                        chunk_file.setframerate(audio_file.getframerate())
                        chunk_start = chunk_index * chunk_length * frame_rate
                        chunk_end = chunk_start + chunk_length * frame_rate
                        chunk_file.writeframes(audio_file.readframes(chunk_end - chunk_start))
    print("ALL AUDIOS TRIMMED SUCCESFULLY")
    

def load_annotations(path_annot, nombre):
    """
        Loads annotations from .txt files in the given directory and returns a dataframe.

    Parameters
    ----------
    path_annot : str
        The path to the folder containing the annotation files.
    nombre : list
        A list of strings representing the names of the annotation files (without the '.txt' extension).

    Returns
    -------
    df_mlabel : pd.DataFrame
        The dataframe containing the loaded annotations.
    fname_lists : list
        A list of filenames corresponding to the loaded annotations.
    """
    df = pd.DataFrame()
    for i in range (len(nombre)):
        flist = glob.glob(path_annot + nombre[i])
        for fname in flist:
            df_aux = util.read_audacity_annot(fname)
            df_aux['fname'] = os.path.basename(fname).replace('.txt', '.wav')
            df_aux = df_aux.drop(columns=['min_f', 'max_f'])
            df = pd.concat([df,df_aux],ignore_index=True)
            df.reset_index(inplace=True, drop=True)
            df_mlabel, fname_lists = filter_window(df,0,110,5)
    return df_mlabel, fname_lists
    
    
# def generate_spectrogram(audio_file_path, output_file_path, sr=22050, n_fft=1723, hop_length=64):
#     """
#     Generate spectrogram from audio file and save it as an image.
    
#     Parameters
#     ----------
#     audio_file_path : str
#         Path to audio file
#     output_file_path : str
#         Path to save the generated spectrogram
#     sr : int, optional
#         Sampling rate of the audio file, by default 22050
#     n_fft : int, optional
#         Length of the FFT window, by default 2048
#     hop_length : int, optional
#         Number of samples between successive frames, by default 512
    
#     Returns
#     -------
#     None
#     """
#     y, sr = librosa.load(audio_file_path, sr=sr)
#     S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
#     log_S = librosa.amplitude_to_db(S)
#     plt.figure(figsize=(8, 8))
#     librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
#     plt.axis('off')
#     plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, dpi=100)
#     plt.close()

def generate_spectrogram(audio_file_path, output_file_path, sr=22050, n_fft=1024, hop_length=64, n_mels=256):
    """
    Generate mel spectrogram from audio file and save it as an image.
    
    Parameters
    ----------
    audio_file_path : str
        Path to audio file
    output_file_path : str
        Path to save the generated spectrogram
    sr : int, optional
        Sampling rate of the audio file, by default 22050
    n_fft : int, optional
        Length of the FFT window, by default 2048
    hop_length : int, optional
        Number of samples between successive frames, by default 512
    n_mels : int, optional
        Number of mel bands to generate, by default 128
    
    Returns
    -------
    None
    """
    y, sr = librosa.load(audio_file_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_S = librosa.amplitude_to_db(S)
    plt.figure(figsize=(8, 8))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', cmap='jet')
    #plt.colorbar(format='%+2.0f dB')
    plt.axis('off')
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


def generate_spectrograms_from_folder(input_folder, output_folder):
    """
    Generates spectrograms from WAV files in a given folder and saves the spectrograms in a specified output folder.

    Parameters
    ----------
    input_folder : str
        The path to the folder containing the WAV files.
    output_folder : str
        The path to the folder where the spectrograms will be saved. If the folder doesn't exist, it will be created.

    Returns
    -------
    None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            input_file_path = os.path.join(subdir, file)
            if input_file_path.endswith('.wav'):
                output_file_path = os.path.join(output_folder, subdir.replace(input_folder, ''), file[:-4] + '.png')
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                generate_spectrogram(input_file_path, output_file_path)  
                
                
def get_spectrogram_paths(directory):
    """
    Return a list of spectrogram file paths found in the directory.

    Parameters
    ----------
    directory : str
        The directory to search for spectrogram files.

    Returns
    -------
    list
        List of spectrogram file paths.
    """
    spectrogram_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                spectrogram_path = os.path.join(root, file)
                spectrogram_paths.append(spectrogram_path)
    return spectrogram_paths


def get_labels_from_csv(csv_file_path):
    """
    This function reads a csv file and returns a list of labels.

    Parameters
    ----------
    csv_file_path : str
        The path to the csv file that contains the labels.

    Returns
    -------
    list
        A list of strings representing the labels.
    """
    df = pd.read_csv(csv_file_path)
    return df['label'].tolist()


def create_spectrogram_dataframe(directory):
    """
    Creates a Pandas DataFrame containing the names and file paths of spectrograms in a directory.
    
    Parameters
    ----------
    directory : str
        The directory containing the spectrogram files.
    
    Returns
    -------
    pandas.DataFrame
        A dataframe containing two columns: 'NAME', the basename of the spectrogram file without the '.png' extension, 
        and 'Path', the file path to the spectrogram.
    """    
    spectrogram_paths = get_spectrogram_paths(directory)
    spectrogram_names = [os.path.basename(path).replace('.png', '') for path in spectrogram_paths]
    spectrogram_df = pd.DataFrame({'NAME': spectrogram_names, 'Path': spectrogram_paths})
    return spectrogram_df


# def create_label_df(input_folder):
#     """
#     Creates a label DataFrame based on csv files in the input folder.

#     Parameters
#     ----------
#     input_folder : str
#         The path to the folder containing the csv files.

#     Returns
#     -------
#     label_df : pandas DataFrame
#         The created label DataFrame containing the labels of the csv files.

#     """
#     label_dict = {
#         'PHYCUV_M': 0,
#         'PHYCUV_F': 0,
#         'BOAALB_M': 0,
#         'BOAALB_F': 0,
#         'BOALUN_F': 0,
#         'BOALUN_M': 0,
#         'PHYCUV_M': 0,
#         'PHYCUV_F': 0
#     }
#     label_df = pd.DataFrame(columns=['NAME'] + list(label_dict.keys()))
    
#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             if file.endswith(".csv"):
#                 csv_file_path = os.path.join(root, file)
#                 labels = get_labels_from_csv(csv_file_path)
#                 file_name = os.path.splitext(file)[0]
#                 label_dict = {key: 1 if key in labels else 0 for key in label_dict}
#                 label_df = label_df.append({'NAME': file_name, **label_dict}, ignore_index=True)
#     return label_df

def create_label_df(input_folder):
    """
    Creates a label DataFrame based on csv files in the input folder.

    Parameters
    ----------
    input_folder : str
        The path to the folder containing the csv files.

    Returns
    -------
    label_df : pandas DataFrame
        The created label DataFrame containing the labels of the csv files.

    """
    label_dict = {
        'PHYCUV_M': 0,
        'PHYCUV_F': 0,
        'BOAALB_M': 0,
        'BOAALB_F': 0,
        'BOALUN_F': 0,
        'BOALUN_M': 0,
        'BOALUN_C': 0,
        'none': 0
    }
    label_df = pd.DataFrame(columns=['NAME'] + list(label_dict.keys()))
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                labels = get_labels_from_csv(csv_file_path)
                file_name = os.path.splitext(file)[0]
                has_labels = any(label in labels for label in label_dict)
                label_dict['none'] = 1 if not has_labels else 0
                label_dict.update({key: 1 if key in labels else 0 for key in label_dict if key != 'none'})
                label_df = label_df.append({'NAME': file_name, **label_dict}, ignore_index=True)
    return label_df

def save_merged_df(merged_df, save_path):
    """
     Saves the dataframes to a csv file.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        Dataframe to be saved.
    save_path : str
        The path to save the csv file.

    Returns
    -------
    None
    """
    file_path = os.path.join(save_path, "merged_df.csv")
    merged_df.to_csv(file_path, index=False)
    



def preprocess_audio_data(df):
    # Define a function to apply to each group
    # Define a function to apply to each group
    def group_function(group):
        # Calculate the number of 4 second intervals in the group
        num_intervals = int(group['max_t'].max() // 5) + 1
        
        # Create a new DataFrame with one row for each 4 second interval
        df_new = pd.DataFrame({'fname': group['fname'].unique()[0],
                            'interval': range(num_intervals),
                            'labels': ''})
        
        # Assign the correct labels to each 4 second interval
        for idx, row in group.iterrows():
            start = int(row['min_t'] // 5)
            end = int(row['max_t'] // 5) + 1
            labels = row['label'].split(',')
            df_new.loc[start:end, 'labels'] = df_new.loc[start:end, 'labels'].apply(lambda x: ','.join(list(set(x.split(',') + labels))))
            
        return df_new
    # Group by filename and apply the group function to each group
    grouped = df.groupby('fname')
    df_new = pd.concat([group_function(group) for name, group in grouped])

    # Reset the index of the new DataFrame
    df_new = df_new.reset_index(drop=True)

    # Add the interval number to the filename
    df_new['fname'] = df_new.apply(lambda row: row['fname'].rsplit('.', 1)[0] + '_' + str(row['interval']) + '.wav', axis=1)

    # Concatenate all labels with commas and split to get a set of unique labels
    unique_labels = set(','.join(df_new['labels'].values.tolist()).split(','))

    # Create new columns for each unique label
    for label in unique_labels:
        # Convert label name to a valid column name by replacing any special characters with underscores
        col_name = label.replace(' ', '_').replace('-', '_').replace('/', '_')

        # Populate the column with 1 if the label is present in the row, 0 otherwise
        df_new[col_name] = df_new['labels'].apply(lambda x: 1 if label in x else 0)
    
    columns_to_drop = ['labels', 'interval', 'PITAZU_F', 'DENCRU_F', 'PHYMAR_C', 'PHYMAR_M', 'DENCRU_M', 'PITAZU_M','PHYMAR_F','nan']
    df_new = df_new.drop(columns_to_drop, axis=1)
    df_new['fname'] = df_new['fname'].str.replace('.wav', '')
    df_new = df_new.rename(columns={'fname': 'NAME'})

    # Save the preprocessed DataFrame to a CSV file
    df_new.to_csv('../NEW_Orig_Data/DATASETS/label_df.csv', index=False)

    return df_new

                
                  
print("_____________DONE LOADING THE FUNCTIONS_____________")
#__________________________END OF FUNCTIONS_____________________________________
#%%

#Load multiple annotations from a directory and perform multi-label window cutting.
df_mlabel,fname_lists = load_annotations(path_annot, nombre)
#Saving fname_lists for tracking purposes      
#save_fname_lists(fname_lists, '../STRETCHED_CSV/')       
print("Done!") 


#%% Merging mlabel list of dataframes to a single dataframe to count numbers of labels  on audios
v = pd.DataFrame()
v = pd.concat(df_mlabel,ignore_index=True)
print("Done counting labels")
# Print count of species founded    
print(v['label'].value_counts())

#%%
#Generar dataset
# Group the DataFrame by file name
label_df = preprocess_audio_data(v)

#%%
#Filtrar la especie y guardar en df_rois
# df_rois = df[df.apply(filter_label, axis=1)].reset_index()


#%% Trimming audio files from src directory
trim_audio_files(src_dir, dst_dir, chunk_length)


#%% Generating spectrogram files and saving on a local file
#Important, no more than 100 audios per batch, this depends on the size of the spectrogram required
generate_spectrograms_from_folder(input_folder, output_folder)


#%%
#Creating spectrogram dataframe from the spectrogram files folder
spectrogram_df = create_spectrogram_dataframe(spectrogram_folder)
spectrogram_df.to_csv('../NEW_Orig_Data/DATASETS/spectrogram_df.csv', index=False)

#%%
#CREATE THE LABELS DATAFRAME IN WHICH IS A SUMMARY OF SPECIES BY NAME OF FILE
#label_df = create_label_df(input_folder_LBL)


#%%
# Merge the two dataframes, 'label_df' & 'spectrogram_df' by the 'NAME' column to get the dataframe for the ML model
merged_df = spectrogram_df.merge(label_df, on='NAME', how='right')


#%%
#Saving DataFrames in the same path
save_path = "../NEW_Orig_Data/DATASETS/"
#save_merged_df(label_df, save_path)
#save_merged_df(spectrogram_df, save_path)
save_merged_df(merged_df, save_path)


# %%

merged_df_TEST = pd.read_csv("../NEW_Orig_Data/DATASETS/merged_df.csv",delimiter=';')
#%%
# Drop the "none" column
merged_df_TEST.drop(columns=['none'], inplace=True)
#%%
# Merge PHYCUV_M and PHYCUV_F columns into a new column called PHYCUV
merged_df_TEST['PHYCUV'] = (merged_df_TEST['PHYCUV_M'].fillna(0) + merged_df_TEST['PHYCUV_F'].fillna(0)).clip(0,1)
#%%
# Merge BOAALB_M and BOAALB_F columns into a new column called BOAALB
merged_df_TEST['BOAALB'] = (merged_df_TEST['BOAALB_M'].fillna(0) + merged_df_TEST['BOAALB_F'].fillna(0)).clip(0,1)

# Merge BOALUN_M and BOALUN_F columns into a new column called BOALUN
merged_df_TEST['BOALUN'] = (merged_df_TEST['BOALUN_M'].fillna(0) + merged_df_TEST['BOALUN_F'].fillna(0)).clip(0,1)
merged_df_TEST['BOALUN'] = (merged_df_TEST['BOALUN'].fillna(0) + merged_df_TEST['BOALUN_C'].fillna(0)).clip(0,1)
# Drop the original male and female columns
merged_df_TEST.drop(columns=['PHYCUV_M','BOAALB_M','BOALUN_M','PHYCUV_F','BOAALB_F','BOALUN_F','BOALUN_C'], inplace=True)
#merged_df_TEST.drop(columns=['PHYCUV'], inplace=True)
# Save the modified dataframe to a CSV file
merged_df_TEST.to_csv("../NEW_Orig_Data/DATASETS/merged_df_3LABELS.csv", index=False)


# %%
