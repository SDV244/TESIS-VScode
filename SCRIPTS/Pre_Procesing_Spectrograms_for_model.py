
import numpy as np
import pandas as pd
import glob
from os import walk
import librosa
import librosa.display
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import tensorflow as tf


# Preprocessing the dataset

df = pd.read_csv('/content/Dataset_Super_Aumented.csv', delimiter=';')
#df = pd.read_csv('../SCRIPTS/TDL/PHYCUV/DATASET/merged_df_personal.csv')
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
#%%

# Images path

#Creating auxiliar Dataframe
df_T = df
# Preprocess images creating caracteristic array
X = preprocess_images(image_paths) 
 # Obtaining labels array in Numpy format
y = np.array(df.drop(['NAME','Path'],axis=1))
#Declaring size of mages
SIZE = 224
# Dividing Dataset in training and testing with 20 percent of whole dataset for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

# verify the distribution of labels in the train and test sets
import numpy as np
train_label_counts = np.sum(y_train, axis=0)
test_label_counts = np.sum(y_test, axis=0)
print(f"Train label counts: {train_label_counts}")
print(f"Test label counts: {test_label_counts}")