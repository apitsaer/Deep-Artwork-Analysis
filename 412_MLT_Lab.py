#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:03:07 2019

@author: admin
"""

import os # used for navigating to image path
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re

from PIL import Image # used for loading images

import numpy as np
from numpy import argmax
import pandas as pd
from random import shuffle
from math import ceil

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 200
DIR = '../data/TestSetA/'
INPUT_FILE = 'AWTableSetA.csv'
BATCH_SIZE = 32
N_EPOCHS = 30

#naming_dict = {} # AW id: artist

def get_IMG_size_statistics(DIR):
    heights = []
    widths = []
    directory = DIR + 'original'
    for img in os.listdir(directory): 
        if(img.endswith('jpg')):
            path = os.path.join(directory, img)
            data = np.array(Image.open(path)) #PIL Image library
            heights.append(data.shape[0])
            widths.append(data.shape[1])
    print(img)
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))
    
def genDataSets(AWTable):
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_valid_index in split.split(AWTable, AWTable.Artist):
        train_set = AWTable.iloc[train_index]
        test_valid_set = AWTable.iloc[test_valid_index]

    #Also build a validation set by splitting the previously generated test set
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for test_index, valid_index in split2.split(test_valid_set, test_valid_set.Artist):
        test_set = test_valid_set.iloc[test_index]
        valid_set = test_valid_set.iloc[valid_index]

    return (train_set, valid_set, test_set)


def learnModelMulti(n_iter, batch_size = 20):

    from keras.models import Model
    from keras import layers
    from keras import Input
    import matplotlib.pyplot as plt
    
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image')
    x = layers.Conv2D(32, kernel_size = (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, kernel_size = (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, kernel_size = (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(96, kernel_size = (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(32, kernel_size = (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x= layers.Dense(128, activation='relu')(x)
        
    artist_prediction = layers.Dense(nClassesArtist, activation='softmax', name='artist')(x)
    type_prediction = layers.Dense(nClassesType, activation='softmax', name='type')(x)
    model = Model(image_input,[artist_prediction, type_prediction])
    model.summary()
        
    model.compile(optimizer='adam',
                  loss={'artist': 'binary_crossentropy','type': 'binary_crossentropy'},
                  loss_weights={'artist': 1, 'type': 1},
                  metrics={'artist': 'accuracy', 'type': 'accuracy'})
    
#    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
#                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
#                             horizontal_flip=True, fill_mode="nearest")
    
    train_gen = data_generator(train_set, IMG_SIZE, IMG_SIZE, batch_size, dataAugm=True)
    valid_gen = data_generator(valid_set, IMG_SIZE, IMG_SIZE, batch_size, dataAugm=False)
    test_gen = data_generator(test_set, IMG_SIZE, IMG_SIZE, batch_size, dataAugm=False)
    
    history = model.fit_generator(train_gen, 
                        validation_data = valid_gen,  
                        validation_steps = ceil(valid_set.shape[0] / batch_size),
                        steps_per_epoch = 2 * ceil(train_set.shape[0] / batch_size),
                        epochs = n_iter, verbose = 1)

    model.save('artistsRD.h5')
 
    type_acc = history.history['type_acc']
    val_type_acc = history.history['val_type_acc']
    type_loss = history.history['type_loss']
    val_type_loss = history.history['val_type_loss']
    artist_acc = history.history['artist_acc']
    val_artist_acc = history.history['val_artist_acc']
    artist_loss = history.history['artist_loss']
    val_artist_loss = history.history['val_artist_loss']
    epochs = range(1, len(artist_acc) + 1)
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(epochs, type_acc, 'bo', label='Type - Train acc')
    axs[0, 0].plot(epochs, val_type_acc, 'ro', label='Type - Val acc')
    axs[0, 0].set_title('Type')
    axs[0, 0].set_ylabel('Acc')
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, artist_acc, 'bo', label='Artist - Train acc')
    axs[0, 1].plot(epochs, val_artist_acc, 'ro', label='Artist - Val acc')
    axs[0, 1].set_title('Artist')
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, type_loss, 'bo', label='Type - Train loss')
    axs[1, 0].plot(epochs, val_type_loss, 'ro', label='Type - Val loss')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()

    axs[1, 1].plot(epochs, artist_loss, 'bo', label='Artist - Train loss')
    axs[1, 1].plot(epochs, val_artist_loss, 'ro', label='Artist - Val loss')
    axs[1, 1].legend()
  
    for ax in axs.flat:
        ax.set(xlabel='Epoch')
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        
    plt.savefig('NEW Train_valid_accuracy')
    plt.show()
    
    loss, artist_loss, type_loss, artist_acc, type_acc = model.evaluate_generator(test_gen,
                                                                                  steps = ceil(test_set.shape[0]/batch_size))
    
    print('model.metrics_names = ' +  ' '.join(model.metrics_names))
    print('test_artist_acc = ' + str(artist_acc))
    print('test_type_acc = ' + str(type_acc))
    
def testModelMulti(batch_size = 20):
    
    model = load_model('artistsRD.h5')
    test_gen = data_generator(test_set, IMG_SIZE, IMG_SIZE, batch_size)
    loss, artist_loss, type_loss, artist_acc, type_acc = model.evaluate_generator(test_gen, 
                                                                                  steps = ceil(test_set.shape[0]/batch_size))
    
    print('model.metrics_names = ' +  ' '.join(model.metrics_names))
    print('test_artist_acc = ' + str(artist_acc))
    print('test_type_acc = ' + str(type_acc))
    

def data_generator(data_set, img_height,img_width, batch_size = 20, dataAugm=False):
       
     dir_img = DIR + 'original'
     iBatch = 0
     steps_per_epoch = ceil(data_set.shape[0] / batch_size)
     
     while True:
          # TO DO: shuffle batch_Id to view training samples in different order
          batch_Id = data_set.iloc[iBatch * batch_size : (iBatch+1) * batch_size].index
          batch_image = []
          batch_target_artist = [] 
          batch_target_type = []
          batch_target_mat = []

          for Id in batch_Id:
              artist = AWTableTOP.loc[Id].Artist
              types = AWTableTOP.loc[Id].Type_all
              mats = AWTableTOP.loc[Id].Material_all
              path = os.path.join(dir_img, Id +".jpg")
              img = Image.open(path)
              img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
              batch_image.append(np.array(img) / 255)
              batch_target_artist.append(artist)
              batch_target_type.append(getTypes(types))
              batch_target_mat.append(getTypes(mats))
              if(dataAugm):
                  # Basic Data Augmentation - Horizontal Flipping
                  flip_img = img
                  flip_img = np.array(flip_img)
                  flip_img = np.fliplr(flip_img)
                  batch_image.append(np.array(flip_img) / 255)
                  batch_target_artist.append(artist)
                  batch_target_type.append(getTypes(types))
                  batch_target_mat.append(getTypes(mats))
          
          batch_image = np.array( batch_image) / 255
          batch_target_artist = np.array( batch_target_artist ).reshape(-1,1)
          batch_target_artist = encoder_Artist.transform(batch_target_artist)

          batch_target_type = encoder_Type.transform(batch_target_type)
          batch_target_mat = encoder_Mat.transform(batch_target_mat)
          
          yield(batch_image, {'artist': batch_target_artist, 'type': batch_target_type} )
          iBatch += 1
          if(iBatch == steps_per_epoch): iBatch = 0
          
def getTypes(text):
    types = []
    text_type = re.split(";", text)
    for typ in text_type:
        if not typ == '':
            types.append(typ)  
    return types

def getAllTypeMat(AWTableTOP):

    all_type = set()
    all_mat = set()
    
    for row in AWTableTOP.itertuples():
        text_mat = re.split(";", row.Material_all)
        for mat in text_mat:
            if not mat == '':
                all_mat.add(mat)
    
        text_type = re.split(";", row.Type_all)
        for typ in text_type:
            if not typ == '':
                all_type.add(typ)
        
    return (all_type, all_mat)
    
################# MAIN

#get_IMG_size_statistics(DIR)
print("*****    1. Generating data sets")
AWTableTOP = pd.read_csv(DIR + INPUT_FILE, keep_default_na=False)
AWTableTOP.set_index("Id", inplace=True)
AWTableTOP.columns = AWTableTOP.columns.str.strip() #remove leading and trailing white space if n
nClassesArtist = AWTableTOP['Artist'].nunique()
nClassesType = AWTableTOP['Type'].nunique()

(all_Types, all_Mats) = getAllTypeMat(AWTableTOP)

print('Found {} uniaue Type'.format(len(all_Types)))
print('Found {} uniaue Material'.format(len(all_Mats)))

(train_set, valid_set, test_set) = genDataSets(AWTableTOP)

# One Hot Encoding - Fitting
encoder_Type = MultiLabelBinarizer()
all_Types = list(all_Types)
all_Types = [[i] for i in all_Types]
encoder_Type.fit(all_Types)

encoder_Mat = MultiLabelBinarizer()
all_Mats = list(all_Mats)
all_Mats = [[i] for i in all_Mats]
encoder_Mat.fit(all_Mats)

encoder_Artist = OneHotEncoder(sparse=False)
all_Artists = AWTableTOP.Artist.unique().reshape(-1,1)
encoder_Artist.fit(all_Artists)

print("*****    2. Learning and evaluating on test set")

#learnModelMulti(N_EPOCHS, BATCH_SIZE)
#testModelMulti(BATCH_SIZE)

# To DO 
# implement full multi task
# implement RESNET 50
# implement callback and logs
# run on full dataset
# 2. Use Keras data augmentation

# try online data augmentation

# implement step decay

# =============================================================================
train_gen = data_generator(train_set, IMG_SIZE, IMG_SIZE, BATCH_SIZE)
# valid_gen = data_generator(valid_set, IMG_SIZE, IMG_SIZE, BATCH_SIZE)
# test_gen = data_generator(test_set, IMG_SIZE, IMG_SIZE, BATCH_SIZE)
# steps_per_epoch = ceil(valid_set.shape[0] / BATCH_SIZE)
#for i in range(1,2*(40+1)):
#    print('*************** Step : ' + str(i))
#    (batch_image, dicto) = next(train_gen)
#    print(dicto.get('type')[0])
# 
# =============================================================================

# def onHotDecode(values):
#     ## invert first example
#     inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
#     print(inverted)
 