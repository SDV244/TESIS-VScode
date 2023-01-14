#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build dataset

Audio recordings come into monophonic recordings. Manual annotations come into 
time-frequency segmentation performed using audacity or raven. There is one annotation
file per audio recording.

1. Transform bounding box annotation data to fixed size samples

"""

import numpy as np  
import pandas as pd
from maad import sound, util
from utils import roi2windowed, find_file
import os
import glob

# Set main variables
target_fs = 24000  # target fs of project
wl = 5  # Window length for formated rois
path_annot = './annotations/'  # location of bbox annotations
path_audio = './audio/'  # location of raw audio
path_save = './train_dataset/'  # location to save new samples

#%% Load multiple annotations from directory
flist = glob.glob(path_annot+'*.txt')
df = pd.DataFrame()
for fname in flist:
    df_aux = util.read_audacity_annot(fname)
    df_aux['fname'] = os.path.basename(fname).replace('.txt', '.wav')
    df = df.append(df_aux)
df.reset_index(inplace=True, drop=True)

#%% Post-process annotations. 
# Select vocalizations of a single species
df_rois = df.loc[(df.label=='BOAPRA_CLR') | (df.label=='BOAPRA_MED'), :]

#%% format rois to a fixed window
rois_fmt = pd.DataFrame()
for idx, roi in df_rois.iterrows():
    roi_fmt = roi2windowed(wl, roi)
    rois_fmt = rois_fmt.append(roi_fmt)

rois_fmt.reset_index(inplace=True, drop=True)
print(rois_fmt)

#%% Load audio, resample, trim, normalize and write to disk
for idx_row, roi_fmt in rois_fmt.iterrows():
    print(idx_row+1, '/', len(rois_fmt))
    full_path_audio = find_file(roi_fmt.fname, path_audio)[0]
    fname_audio = os.path.join(full_path_audio)
    s, fs = sound.load(fname_audio)
    s = sound.resample(s, fs, target_fs)
    s_trim = sound.trim(s, target_fs, roi_fmt.min_t, roi_fmt.max_t, pad=True)

    fname_save = os.path.join(path_save, roi_fmt.label+'_'+str(idx_row).zfill(3)) 
    sound.write(fname_save+'.wav', fs=target_fs, data=s_trim)

# save data frame
rois_fmt['fname_trim'] = rois_fmt.label + '_' + rois_fmt.index.astype(str).str.zfill(3)
rois_fmt.to_csv(path_save+'rois_details.csv', index=False)
