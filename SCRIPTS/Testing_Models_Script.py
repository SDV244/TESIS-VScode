#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from maad import sound, util
import glob
from os import walk
import sphinx
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import wave
from tensorflow.keras.models import Sequential
import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import PIL as image_lib
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import time
from tensorflow.keras.models import load_model
#Dataframe for training 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# Preprocessing the dataset

#df = pd.read_csv('/content/TESIS-VScode/SCRIPTS/TDL/PHYCUV/DATASET/Datos_30_comprobacion_para_augVAL.csv')
#df = pd.read_csv('/content/TESIS-VScode/New_Data_Stretched/STRETCHED_DATASET/Datos_30_para_probar_modelos.csv')
df = pd.read_csv('/content/TESIS-VScode/NEW_Orig_Data/DATASETS/Datos_30_para_probar_modelos.csv')
df.dropna(subset=['Path'], inplace=True)
#%%
#Function for preprocesing images
def preprocess_images(paths, target_size=(224,224,3)):
    X = []
    for path in paths:
        path = os.path.abspath(path)
        path = path.replace('\\', '/')
        img = image.load_img(path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array/255     
        X.append(img_array)
    return np.array(X)
image_paths = df['Path'].values




X = preprocess_images(image_paths) 
y = np.array(df.drop(['NAME','Path'],axis=1))

#%%

import os
import pandas as pd
from tensorflow.keras.models import load_model

# Directory where the models are stored
model_dir = "/content/drive/MyDrive/Models/TMSK"

# CSV file containing the dataset
# List to store the results
results = []

# Loop through all files in the model directory
for root, dirs, files in os.walk(model_dir):
    for file in files:
        # Check if file is a Keras model file
        if file.endswith(".h5"):
            # Load the model
            model_path = os.path.join(root, file)
            model = load_model(model_path)           
            # Make predictions on the test data
            y_pred = model.predict(X)
            
            # Compute the evaluation metrics
            test_f1_score = f1_score(y, y_pred > 0.5, average=None)
            
            # Append the results to the list
            results.append({
                "model_name": file,
                "f1_score": test_f1_score
            })

# Create a dataframe from the results list
df_results = pd.DataFrame(results)

# Save the dataframe to Excel
df_results.to_excel("model_results.xlsx", index=False)

