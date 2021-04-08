from keras.models import Sequential, Input
from keras.layers import LeakyReLU, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from functools import partial
import numpy as np
import keras as k
import glob
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
from keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from keras.models import Model

import seaborn as sns
from sklearn.metrics import classification_report


import gc
import torch

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


#################################################################################
import pickle
filename = "C:/Users/MSE/Desktop/Tez Çalışması Yeni/Adaptive_Histogram_Equalization/AHE-Left-images/Ahe(Clip_0,1-Disk_4).obj"
#Save
#pickle.dump(Df, open(filename, 'wb'))
#Load
Df = pickle.load( open( filename, "rb" ) )

x_train, x_test, y_train, y_test = train_test_split(Df.iloc[:, 0:-1], Df.iloc[:, -1], test_size=0.20, random_state=0)
X_train = x_train.values.reshape(-1, 224, 112, 1)
X_test = x_test.values.reshape(-1, 224, 112, 1)
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

test = X_train[1,:]

#plt.imshow(test)

#plt.gray()
################################################################################

Optimizer = Adam(lr=0.80e-5)
new_conv = partial(Conv2D ,padding = "same")

def _base_block(out,x):
    "(3,3), Leaky, Batch"
    x =new_conv(out, (3,3))(x)
    x =LeakyReLU(alpha=0.1)(x)
    x =BatchNormalization()(x)
    return x

def _block_1(out, x):
    """
    output follows:
    out//2, out

    """
    x =  new_conv(out//2, (1,1))(x)
    x =LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = _base_block(out,x)
    return x

def _block_2(out, x):
    """
    output follows:
    out, out//2, out

    """
    x =_base_block(out,x)
    x = _block_1(out, x)
    return x


def Darknet19():
    input_layer = Input((img_size, 112, 1))
    x = _base_block(32,input_layer)
    x =  MaxPooling2D((2,2),strides = 2)(x)
    x = _base_block(64,x)
    x = MaxPooling2D((2,2),strides = 2)(x)
    x = _block_2(128, x)
    x = MaxPooling2D((2,2),strides = 2)(x)
    x = _block_2(256, x)
    x = MaxPooling2D((2,2),strides = 2)(x)
    x = _block_2(512, x)
    x = _block_1(512, x)
    x = MaxPooling2D((2,2),strides = 2)(x)
    x =_block_2(1024, x)
    x = _block_1(512, x)
    x = new_conv(3, (1,1), activation = "linear")(x)
    model = Model(inputs = input_layer, outputs = x)
    return model

def Darknet_classifier():
    base_model = Darknet19()
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Activation("softmax")(x)
    model = Model(inputs = base_model.inputs, outputs = output)

    return model

img_size = 224 #multiple of 30

#print(model.summary())


kernel_size = [4,8,16,28,32,56,112]
clip_limit = [1]

for j in clip_limit:

    gc.collect()
    torch.cuda.empty_cache()
    for i in kernel_size:

        filename = "C:/Users/MSE/Desktop/Tez Çalışması Yeni/Adaptive_Histogram_Equalization/AHE-Right-images/Ahe-Right(Clip_%s" %j + "-Disk_%s).obj" %i
        Df = pickle.load( open( filename, "rb" ) )
        x_train, x_test, y_train, y_test = train_test_split(Df.iloc[:, 0:-1], Df.iloc[:, -1], test_size=0.20, random_state=0)
        X_train = x_train.values.reshape(-1, 224, 112, 1)
        X_test = x_test.values.reshape(-1, 224, 112, 1)
        Y_train = to_categorical(y_train)
        Y_test = to_categorical(y_test)


        model = Darknet19()
        model =Darknet_classifier()

        model.compile(optimizer = Optimizer , loss = 'categorical_crossentropy' , metrics=['accuracy'])

        #model_chkpt = ModelCheckpoint('Adaptive_Histogram_Equalization/Model-h5_DarkNet/DarkNet(Clip_0,%s' %j + '-Disk_%s).h5' %i, save_best_only=True, monitor='accuracy')

        #early_stopping = EarlyStopping(monitor='loss', restore_best_weights=False, patience=10)

        history = model.fit(X_train, Y_train,
                            validation_split=0.20,
                            epochs=40, batch_size=10, shuffle=True,
                           #callbacks=[model_chkpt, early_stopping]
                            )

        fig, ax = plt.subplots(1, 2, figsize=(12, 3))
        ax[0].plot(history.history['loss'], color='b', label="Training loss")
        ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        legend = ax[0].legend(loc='best', shadow=True)
        ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
        ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
        legend = ax[1].legend(loc='best', shadow=True)
        ax = plt.subplot()
        ax.set_title('Confusion Matrix -- DarkNet_Right(Ahe-%s'%j+'-%s)' %i)
        pred = model.predict(X_test)
        Y_TEST = np.argmax(Y_test, axis=1)
        cm = metrics.confusion_matrix(Y_TEST, np.argmax(pred, axis = 1))
        classes = ['normal', 'other pneumonia', 'covid19']
        sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('Adaptive_Histogram_Equalization/Results_DarkNet_Right/DarkNet_Right(Ahe-%s'%j+'-%s-conf)' %i)
        plt.show

        plt.close()

        report = classification_report(Y_TEST, np.argmax(pred, axis = 1), digits=4, output_dict=True)
        ax = plt.axes()
        ax = sns.heatmap(pd.DataFrame(report).iloc[:, :].T, annot=True, fmt=".5g", linewidths=.4, ax = ax, cbar=False)
        ax.set_title('normal = 0 , other pneumonia = 1, covid19 = 2')
        plt.xlabel('DarkNet_Right(Ahe-%s'%j+'-%s)' %i, fontsize = 8)
        plt.savefig('Adaptive_Histogram_Equalization/Results_DarkNet_Right/DarkNet_Right(Ahe-%s'%j+'-%s-report)' %i)
        plt.show()
        model = None
        del X_test
        del x_test
        del y_test
        del Y_test
        del X_train
        del y_train
        del Y_train
        del report
      #  del model_chkpt
        gc.collect()
        torch.cuda.empty_cache()

    model = None
    gc.collect()
    torch.cuda.empty_cache()


model = None

gc.collect()
torch.cuda.empty_cache()


'''
report = classification_report(Y_TEST, np.argmax(pred, axis = 1), digits=4, output_dict=True)

    from sklearn.metrics import classification_report
    print(classification_report(Y_TEST, np.argmax(pred, axis = 1), digits=4))
    print('normal = 0 , other pneumonia = 1, covid19 = 2')

def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('        ')
        row['class'] = row_data[0]
        row['precision'] = row_data[1]
        row['recall'] = row_data[2]
        row['f1_score'] = row_data[3]
        row['support'] = row_data[4]
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report)
    dataframe.to_csv('classification_report1.csv', index = True)

classifaction_report_csv(report)





ax = plt.axes()
ax = sns.heatmap(pd.DataFrame(report).iloc[:, :].T, annot=True, fmt=".5g", linewidths=.5, ax = ax, cbar=False)
ax.set_title('normal = 0 , other pneumonia = 1, covid19 = 2')
plt.xlabel('AlexNet(Ahe-0,%s' %i + '-%s)' %i, fontsize = 10)
plt.show()
'''






