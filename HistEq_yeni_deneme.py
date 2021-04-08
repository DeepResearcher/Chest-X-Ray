# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 14:11:58 2020

@author: MSE
"""


import keras as k
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
from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
import tensorflow as tf
import time

start = time.time()
print("hello")
end = time.time()
print(end - start)

Disk = 20

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


dim = (224, 224, 3)
dir_name = 'Left-Right_Images/Covid-19'
img_list = glob.glob(dir_name + '/*')


def histEq(dim,dir_name,img_list,Disk):

        list_covid = []
        for img in img_list:
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_eq = rank.equalize(img, selem=disk(80))
            img_eq = rank.equalize(img_eq, selem=disk(60))
            img_eq = rank.equalize(img_eq, selem=disk(40))
            gary2rgb = cv2.cvtColor(img_eq,cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(gary2rgb,(224,224))
            temp_img_array = img_to_array(resized) / 255.0
            list_covid.append(temp_img_array)
        list_covid = np.array(list_covid)
        list_covid2 = list_covid.reshape(-1, 150528)
        df_covid = pd.DataFrame(list_covid2)
        df_covid['label'] = np.full(df_covid.shape[0], 2)
        return df_covid

start = time.time()
df_covid = histEq(dim,dir_name,img_list,Disk)
end = time.time()
time = end - start



list_covid = []
for img in img_list:
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    selem = disk(Disk)
    img_eq = rank.equalize(img, selem=selem)
    gary2rgb = cv2.cvtColor(img_eq,cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(gary2rgb,(224,224))
    temp_img_array = img_to_array(resized) / 255.0
    list_covid.append(temp_img_array)
list_covid = np.array(list_covid)
list_covid2 = list_covid.reshape(-1, 150528)
df_covid = pd.DataFrame(list_covid2)
df_covid['label'] = np.full(df_covid.shape[0], 2)
print(df_covid.shape)



dim = (224, 224, 3)
dir_name2 = 'Left-Right_Images/Normal_L-R'
img_list2 = glob.glob(dir_name2 + '/*')
list_normal = []
for img in img_list2:
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    selem = disk(Disk)
    img_eq = rank.equalize(img, selem=selem)
    gary2rgb = cv2.cvtColor(img_eq,cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(gary2rgb,(224,224))
    temp_img_array = img_to_array(resized) / 255.0
    list_normal.append(temp_img_array)
list_normal = np.array(list_normal)
list_normal2 = list_normal.reshape(-1, 150528)
df_normal = pd.DataFrame(list_normal2)
df_normal['label'] = np.full(df_normal.shape[0], 0)
print(df_normal.shape)



dim = (224, 224, 3)
dir_name3 = 'Left-Right_Images/Pneumonia_L-R'
img_list3 = glob.glob(dir_name3 + '/*')
list_others = []
for img in img_list3:
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    selem = disk(Disk)
    img_eq = rank.equalize(img, selem=selem)
    gary2rgb = cv2.cvtColor(img_eq,cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(gary2rgb,(224,224))
    temp_img_array = img_to_array(resized) / 255.0
    list_others.append(temp_img_array)
list_others = np.array(list_others)
list_others2 = list_others.reshape(-1, 150528)
df_others = pd.DataFrame(list_others2)
df_others['label'] = np.full(df_others.shape[0], 1)
print(df_others.shape)


Df = pd.concat([df_covid, df_normal, df_others], ignore_index=True)



import pickle
filename = "Disk(%s).obj"  % Disk
#Save
pickle.dump(Df, open(filename, 'wb'))
#Load
#Df = pickle.load( open( filename, "rb" ) )


#########################################################################################################################################

f = plt.figure(figsize=(15, 7))
f.suptitle('COVID19', fontsize=20)
f.subplots_adjust(top=2.35)

list_others = []
selem = disk(40)
img = cv2.imread(img_list[5])
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_lhe = rank.equalize(img_gray, selem=selem)
#gary2rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
#resized = cv2.resize(gary2rgb,(224,224))
#temp_img_array = img_to_array(resized) / 255.0
#list_others.append(temp_img_array)
#list_others = np.array(list_others)
#list_others2 = list_others.reshape(-1, 150528)
plt.imshow(glob_adap)
plt.gray()
plt.show()



glob = exposure.equalize_hist(img_gray,nbins=256,mask=True)
glob_adap = exposure.equalize_adapthist(img_gray, clip_limit=0.05, nbins=256)
glob_hist = exposure.adjust_gamma(img_gray,2)


img_gray = rank.equalize(img_gray, selem=selem)
gary2rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
resized = cv2.resize(gary2rgb,(224,224))
temp_img_array = img_to_array(resized) / 255.0
#list_others.append(temp_img_array)
#list_others = np.array(list_others)
#list_others2 = list_others.reshape(-1, 150528)
plt.imshow(temp_img_array)
plt.gray()
plt.show()

print('Resized Dimensions : ',resized.shape)


deneme2 = deneme.reshape(224,224,3)



dim = (224, 224)
dir_name3 = 'Left-Right_Images/Pneumonia_L-R'
img_list3 = glob.glob(dir_name3 + '/*')
list_others = []
for img in img_list3:
    img = cv2.imread(img_list[5])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    selem = disk(Disk)
    img_eq = rank.equalize(img, selem=selem)
    gary2rgb = cv2.cvtColor(img_eq,cv2.COLOR_GRAY2RGB)
    gary2rgb = cv2.resize(gary2rgb,dim)
    temp_img_array = img_to_array(gary2rgb) / 255
    list_others.append(temp_img_array)
list_others = np.array(list_others)
list_others2 = list_others.reshape(-1, 150528)
df_others = pd.DataFrame(list_others2)
df_others['label'] = np.full(df_others.shape[0], 1)
print(df_others.shape)


list_covid = []
for img in img_list:
    temp_img = load_img(img_list[5], grayscale=False, target_size=(224,224,3))
    temp_img_array = img_to_array(temp_img) / 255
    list_covid.append(temp_img_array)
list_covid = np.array(list_covid)
list_covid2 = list_covid.reshape(-1, 150528)
df_covid = pd.DataFrame(list_covid2)
df_covid['label'] = np.full(df_covid.shape[0], 2)



list_others = []
selem1 = disk(20)
selem2 = disk(40)
selem3 = disk(60)
img = cv2.imread(img_list[5])
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_lhe = rank.equalize(img_gray, selem=selem1)
img_lhe2 = rank.equalize(img_lhe, selem=selem2)


img_lhe3 = rank.equalize(img_gray, selem=disk(40),mask=(0.9))

#gary2rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
#resized = cv2.resize(gary2rgb,(224,224))
#temp_img_array = img_to_array(resized) / 255.0
#list_others.append(temp_img_array)
#list_others = np.array(list_others)
#list_others2 = list_others.reshape(-1, 150528)
plt.imshow(img_lhe3)
plt.gray()
plt.show()





artis = 30

img = cv2.imread(img_list[7])
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
f = plt.figure(figsize=(5, 3))
#f.suptitle('Covid-19', fontsize=10)
f.subplots_adjust(top=3.0)
for i in range(1):
    sp = f.add_subplot(1, 1, i + 1)
    selem = disk(artis)
    img_gray1 = rank.equalize(img_gray, selem=selem)
    img_gray = img_gray1
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    plt.imshow(img_gray)
    plt.title('Disk = %s' %artis,fontsize=14)
    artis = artis-20
    plt.gray()
plt.show()





start = time.time()
ilk, artis, azalis = 90, -20, -60
artis1 = np.str(ilk+artis)
ilk1 = np.str(ilk)
azalis1 = np.str(ilk+artis-azalis)
deger = ilk1+"-"+artis1+"-"+azalis1

img = cv2.imread(img_list[7])
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

selem = disk(ilk)
img_lhe = rank.equalize(img_gray, selem=disk(ilk))

img_lhe1 = rank.equalize(img_lhe, selem=disk(ilk+artis))

img_lhe2 = rank.equalize(img_lhe1, selem=disk(ilk+artis-azalis))
end = time.time()

f = plt.figure(figsize=(5, 3))
f.suptitle('Time = %s' %(end - start), fontsize=8)
plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
plt.imshow(glob_adap)
plt.title('Disk = %s' %deger,fontsize=10)
#plt.suptitle('Time = %s' %(end - start))
plt.gray()





deger = 0
for i in range(50):
    img = cv2.imread(img_list[7])
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    start = time.time()
    glob_adap = exposure.equalize_adapthist(img_gray, clip_limit=deger, kernel_size= 90)
    end = time.time()
    f = plt.figure(figsize=(5, 3))
    f.suptitle('Time = %s' %(end - start), fontsize=8)
    plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
    plt.imshow(glob_adap)
    plt.title('Clip_limit = %s' %deger,fontsize=10)
    #plt.suptitle('Time = %s' %(end - start))
    plt.gray()
    deger = deger+0.01




deger = 0
for i in range(10):
    img = cv2.imread(img_list[7])
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    start = time.time()
    glob_adap = rank.equalize(img_gray, selem=disk(deger))
    end = time.time()
    f = plt.figure(figsize=(5, 3))
    f.suptitle('Time = %s' %(end - start), fontsize=8)
    plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
    plt.imshow(glob_adap)
    plt.title('Disk = %s' %deger,fontsize=10)
    #plt.suptitle('Time = %s' %(end - start))
    plt.gray()
    deger = deger+1


deger = 1
glob_adap = exposure.equalize_adapthist(img_gray, clip_limit=deger)



dim = (224, 224, 3)
dir_name = 'Left-Right_Images/Covid-19'
img_list = glob.glob(dir_name + '/*')

dim = (224, 224)
f = plt.figure()
deger = 4
for i in range(1):
    img = cv2.imread(img_list[7])
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    start = time.time()
    #img_gray = cv2.resize(img_gray,dim)
    #lhe = rank.equalize(img_gray, selem=disk(deger))
    glob_adap = exposure.equalize_adapthist(img_gray, clip_limit=0.2, kernel_size= deger)
    end = time.time()
    f = plt.figure(figsize=(5, 3))
    f.suptitle('Time = %s' %(end - start), fontsize=8)
    plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
    #ax = f.add_subplot(1, 2, 1)
    #plt.imshow(lhe)
    #ax = f.add_subplot(1, 2, 2)
    #plt.axis("off")
    plt.imshow(glob_adap)
    #plt.axis("off")
    plt.title('Kernel_size = %s' %deger,fontsize=10)
    #plt.suptitle('Time = %s' %(end - start))
    plt.gray()
    deger = deger+4
