
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



img_size = (224, 224, 3)
dir_name = 'C:/Users/MSE/Desktop/Tez Çalışması Yeni/Left-Right_Images/Covid19_Left'
from os.path import join
from glob import glob
files = []
for ext in ('*.jpeg', '*.png', '*.jpg'):
   files.extend(glob(join(dir_name + '/', ext)))
print(files)
list_covid = []
for img in files:
    temp_img = load_img(img, grayscale=False, target_size=(img_size))
    temp_img_array = img_to_array(temp_img) / 255
    list_covid.append(temp_img_array)
list_covid = np.array(list_covid)
list_covid2 = list_covid.reshape(-1, 150528)
df_covid = pd.DataFrame(list_covid2)
df_covid['label'] = np.full(df_covid.shape[0], 2)



img_size = (224, 224, 3)
dir_name2 = 'Left-Right_Images/Normal_Left'
img_list2 = glob(dir_name2 + '/*')

list_normal = []
for img in img_list2:
    temp_img = load_img(img, grayscale=False, target_size=(img_size))
    temp_img_array = img_to_array(temp_img) / 255
    list_normal.append(temp_img_array)
list_normal = np.array(list_normal)
list_normal2 = list_normal.reshape(-1, 150528)
df_normal = pd.DataFrame(list_normal2)
df_normal['label'] = np.full(df_normal.shape[0], 0)




img_size = (224, 224,3)
dir_name3 = 'Left-Right_Images/Pneumonia_Left'
img_list3 = glob(dir_name3 + '/*')

list_others = []
for img in img_list3:
    temp_img = load_img(img, grayscale=False, target_size=(img_size))
    temp_img_array = img_to_array(temp_img) / 255
    list_others.append(temp_img_array)
list_others = np.array(list_others)
list_others2 = list_others.reshape(-1, 150528)
df_others = pd.DataFrame(list_others2)
df_others['label'] = np.full(df_others.shape[0], 1)


Df = pd.concat([df_covid, df_normal, df_others], ignore_index=True)



import pickle
filename = "Left(WLHE).obj"  
#Save
pickle.dump(Df, open(filename, 'wb'))





