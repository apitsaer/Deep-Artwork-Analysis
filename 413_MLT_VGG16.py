#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:03:07 2019

@author: admin
"""

import os # used for navigating to image path
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re

import numpy as np
from numpy import argmax
import pandas as pd
from random import shuffle
from math import ceil

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

import keras
from keras.models import Model, load_model
from keras import Input, layers, callbacks
from keras.preprocessing import image
from keras.applications import  VGG16

import matplotlib.pyplot as plt

import CustomMetrics as cm

import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 200
DIR = '../../data/TestSetA/'
INPUT_FILE = 'AWTableSetA.csv'
BATCH_SIZE = 20
N_EPOCHS = 100
OUTPUT_PLOTS = 'VGG16'

# get some statistics on the original image size in order to determine acceptable resizing
def get_IMG_size_statistics(DIR):
    heights = []
    widths = []
    directory = DIR + 'original'
    for img in os.listdir(directory): 
        if(img.endswith('jpg')):
            path = os.path.join(directory, img)
            data = image.load_img(path)
            data = image.img_to_array(data)
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
    
# generating training, validation and test data sets with stratified shuffle split
# using Artists as key attribute for the stratification    
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
    
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image')
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))(image_input)
    
#    x = layers.Conv2D(32, kernel_size = (3, 3), activation='relu')(image_input)
#    x = layers.MaxPooling2D(pool_size=(2,2))(x)
#    x = layers.BatchNormalization()(x)
#    
#    x = layers.Conv2D(64, kernel_size = (3, 3), activation='relu')(x)
#    x = layers.MaxPooling2D(pool_size=(2,2))(x)
#    x = layers.BatchNormalization()(x)
#    
#    x = layers.Conv2D(64, kernel_size = (3, 3), activation='relu')(x)
#    x = layers.MaxPooling2D(pool_size=(2,2))(x)
#    x = layers.BatchNormalization()(x)
#
#    x = layers.Conv2D(96, kernel_size = (3, 3), activation='relu')(x)
#    x = layers.MaxPooling2D(pool_size=(2,2))(x)
#    x = layers.BatchNormalization()(x)
#
#    x = layers.Conv2D(32, kernel_size = (3, 3), activation='relu')(x)
#    x = layers.MaxPooling2D(pool_size=(2,2))(x)
#    x = layers.BatchNormalization()(x)
#    
#    x = layers.Dropout(0.2)(x)
    
    x = layers.Flatten()(conv_base)
    x= layers.Dense(256, activation='relu')(x)
        
    artist_prediction = layers.Dense(nClassesArtist, activation='softmax', name='artist')(x)
    year_prediction = layers.Dense(1, name='year')(x) # regression hence no activation function
    type_prediction = layers.Dense(nClassesType, activation='sigmoid', name='type')(x)
    mat_prediction = layers.Dense(nClassesMat, activation='sigmoid', name='mat')(x)

    model = Model(image_input,[artist_prediction, year_prediction, type_prediction, mat_prediction])
    model.summary()
    
    print('This is the number of trainable weights '
          'before freezing the conv base:', len(model.trainable_weights))
    
    conv_base.trainable = False

    print('This is the number of trainable weights '
          'after freezing the conv base:', len(model.trainable_weights))
        
    model.summary()

    model.compile(optimizer='adam',
                  loss={'artist': 'categorical_crossentropy', 'year': 'mae', 'type': 'binary_crossentropy', 'mat': 'binary_crossentropy'},
                  loss_weights={'artist': 1, 'year': 1, 'type': 1, 'mat': 1},
                  metrics={'artist': 'accuracy', 'year': 'mae', 'type': cm.precision, 'mat': cm.precision})
    
#   Code to use Keras built in data augm
#    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
#                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
#                             horizontal_flip=True, fill_mode="nearest")
    
    train_gen = data_generator(train_set, IMG_SIZE, IMG_SIZE, batch_size, dataAugm=True)
    valid_gen = data_generator(valid_set, IMG_SIZE, IMG_SIZE, batch_size, dataAugm=False)
    test_gen = data_generator(test_set, IMG_SIZE, IMG_SIZE, batch_size, dataAugm=False)
    
    callbacks = [keras.callbacks.TensorBoard(log_dir='../logs',
                                             histogram_freq=1,
                                             embeddings_freq=1)]    

    history = model.fit_generator(train_gen, 
                        validation_data = valid_gen,  
                        validation_steps = ceil(valid_set.shape[0] / batch_size),
                        steps_per_epoch = 2 * ceil(train_set.shape[0] / batch_size),
                        epochs = n_iter, verbose = 1)
                        #callbacks=callbacks)

    model.save('artistsRD.h5')
 
    type_precision = history.history['type_precision']
    val_type_precision = history.history['val_type_precision']
    type_loss = history.history['type_loss']
    val_type_loss = history.history['val_type_loss']

    mat_precision = history.history['mat_precision']
    val_mat_precision = history.history['val_mat_precision']
    mat_loss = history.history['mat_loss']
    val_mat_loss = history.history['val_mat_loss']

    year_mae = history.history['year_mean_absolute_error']
    val_year_mae = history.history['val_year_mean_absolute_error']
    year_loss = history.history['year_loss']
    val_year_loss = history.history['val_year_loss']

    artist_acc = history.history['artist_acc']
    val_artist_acc = history.history['val_artist_acc']
    artist_loss = history.history['artist_loss']
    val_artist_loss = history.history['val_artist_loss']
    epochs = range(1, len(artist_acc) + 1)
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(epochs, artist_acc, 'b', label='Train')
    axs[0, 0].plot(epochs, val_artist_acc, 'r', label='Val')
    axs[0, 0].set_title('Artist')
    axs[0, 1].set_ylabel('Acc')
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, year_mae, 'b', label='Train')
    axs[0, 1].plot(epochs, val_year_mae, 'r', label='Val')
    axs[0, 1].set_title('Year')
    axs[0, 1].set_ylabel('MAE')
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, type_precision, 'b', label='Train')
    axs[1, 0].plot(epochs, val_type_precision, 'r', label='Val')
    axs[0, 1].set_title('Type')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].legend()

    axs[1, 1].plot(epochs, mat_precision, 'b', label='Train')
    axs[1, 1].plot(epochs, val_mat_precision, 'r', label='Val')
    axs[0, 1].set_title('Material')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 1].legend()
  
    for ax in axs.flat:
        ax.set(xlabel='Epoch')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        
    plt.savefig('OUTPUT_PLOTS')
    plt.show()
    
def testModelMulti(batch_size = 20):
    
    model = load_model('artistsRD.h5', custom_objects={'precision': cm.precision})

    test_gen = data_generator(test_set, IMG_SIZE, IMG_SIZE, batch_size)
    
    loss, artist_loss, year_loss, type_loss, mat_loss, artist_acc, year_mean_absolute_error, type_precision, mat_precision = model.evaluate_generator(test_gen, 
                                                                                  steps = ceil(test_set.shape[0]/batch_size))
    
    print('model.metrics_names = ' +  ' '.join(model.metrics_names))
    print('test_artist_acc = {:.2%} \t\t' .format(artist_acc))
    print('test_year_mean_absolute_error = {:.2f} \t'.format(year_mean_absolute_error))
    print('test_type_precision = {:.2%} \t\t'.format(type_precision))
    print('test_mat_precision = {:.2%} \t\t'.format(mat_precision))

# Custom data generator to provide batch of labelized data
def data_generator(data_set, img_height,img_width, batch_size = 20, dataAugm=False):
       
     dir_img = DIR + 'original'
     iBatch = 0
     steps_per_epoch = ceil(data_set.shape[0] / batch_size)
     
     while True:
          # TO DO: shuffle batch_Id to view training samples in different order
          batch_Id = data_set.iloc[iBatch * batch_size : (iBatch+1) * batch_size].index
          batch_image = []
          batch_target_artist = [] 
          batch_target_year = []
          batch_target_type = []
          batch_target_mat = []

          for Id in batch_Id:
              artist = AWTableTOP.loc[Id].Artist
              year = AWTableTOP.loc[Id].Year_Est
              types = AWTableTOP.loc[Id].Type_all
              mats = AWTableTOP.loc[Id].Material_all
              path = os.path.join(dir_img, Id +".jpg")
              img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
              img = np.array(img)
              #img = Image.open(path)
              #img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
              batch_image.append(img)
              batch_target_artist.append(artist)
              batch_target_year.append(year)
              batch_target_type.append(stringToList(types))
              batch_target_mat.append(stringToList(mats))
              if(dataAugm):
                  # Basic Data Augmentation - Horizontal Flipping
                  flip_img = img
                  flip_img = np.array(flip_img)
                  flip_img = np.fliplr(flip_img)
                  batch_image.append(np.array(flip_img))
                  batch_target_artist.append(artist)
                  batch_target_year.append(year)
                  batch_target_type.append(stringToList(types))
                  batch_target_mat.append(stringToList(mats))
          
          batch_image = np.array( batch_image) / 255
          batch_target_artist = np.array( batch_target_artist ).reshape(-1,1)
          batch_target_artist = encoder_Artist.transform(batch_target_artist)

          batch_target_year = np.array( batch_target_year ).reshape(-1,1)

          batch_target_type = encoder_Type.transform(batch_target_type)

          batch_target_mat = encoder_Mat.transform(batch_target_mat)
          
          yield(batch_image, {'artist': batch_target_artist, 'year': batch_target_year,'type': batch_target_type, 'mat': batch_target_mat} )
          iBatch += 1
          if(iBatch == steps_per_epoch): iBatch = 0
          
def stringToList(text):
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
AWTableTOP.columns = AWTableTOP.columns.str.strip() #remove leading and trailing white space if any

# creating Year data field with integer equal to mean value of interval
AWTableTOP.insert(3, 'Year_Est', 0)
AWTableTOP.Year_Est = AWTableTOP.Year_Est.astype(float)
for idx, row in AWTableTOP.iterrows():
    year_string = row.Year
    year_split = [int(s) for s in year_string.split() if s.isdigit()]
    if(len(year_split) == 1): year_clean = year_split[0]
    if(len(year_split) == 2): year_clean = (year_split[0] + year_split[1]) / 2
    else: year_clean = 1700 # TO DO : alos throw an exception
    AWTableTOP.set_value(idx, 'Year_Est', year_clean)

(all_Types, all_Mats) = getAllTypeMat(AWTableTOP)
nClassesArtist = AWTableTOP['Artist'].nunique()
nClassesType = len(all_Types)
nClassesMat = len(all_Mats)

print('Found {} unique Type'.format(nClassesType))
print('Found {} unque Material'.format(nClassesMat))

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

learnModelMulti(N_EPOCHS, BATCH_SIZE)
testModelMulti(BATCH_SIZE)

# To DO 
# Artist in OmniArt: The evaluation block for this task contains a softmax layer and class-wise weight matrix for unbalanced data splits
# implement full multi task
# implement RESNET 50
# implement callback and logs
# run on full dataset
# 2. Use Keras data augmentation

# try online data augmentation

# implement step decay


# question
# year transform to range 0 to 1 ?

# ======== CODE to DEBUG the generator =======================================
#train_gen = data_generator(train_set, IMG_SIZE, IMG_SIZE, BATCH_SIZE)
# valid_gen = data_generator(valid_set, IMG_SIZE, IMG_SIZE, BATCH_SIZE)
# test_gen = data_generator(test_set, IMG_SIZE, IMG_SIZE, BATCH_SIZE)
# steps_per_epoch = ceil(valid_set.shape[0] / BATCH_SIZE)
#for i in range(1,2*(40+1)):
#    print('*************** Step : ' + str(i))
#    (batch_image, dicto) = next(train_gen)
#    print(dicto.get('type')[1])
# 
# =============================================================================

# def onHotDecode(values):
#     ## invert first example
#     inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
#     print(inverted)
 
