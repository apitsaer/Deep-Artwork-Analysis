#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:17:55 2019

Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

@author: admin
"""

import numpy as np
import keras
import sklearn.utils
import re

import matplotlib

from keras import applications
from PIL import Image

from random import randint

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
#    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
#                 n_classes=10, shuffle=True):
    def __init__(self, data_set, dir_img, encoder_Artist, encoder_Type, encoder_Mat,
                 year_scaler, batch_size=50, dataAug = False, shuffle=True):
        'Initialization'
        self.data_set = data_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dir_img = dir_img
        self.encoder_Artist = encoder_Artist
        self.encoder_Mat = encoder_Mat
        self.encoder_Type = encoder_Type
        self.year_scaler = year_scaler
        self.dataAug = dataAug
#        self.epochRef = 0
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.data_set) / self.batch_size))
    ## TO DO: TRY floor instead of ceil

    def on_epoch_end(self):
        'Shuffle data set after each epoch'

#        self.epochRef = randint(0, 10000)

        if self.shuffle == True:
            self.data_set = sklearn.utils.shuffle(self.data_set, random_state = 42)

    def __getitem__(self, index):
        'Generate one batch of data'

        batchSubset = self.data_set.iloc[index * self.batch_size : min((index+1) * self.batch_size, self.data_set.shape[0]), :]

        batch_target_artist = [] 
        batch_target_year = []
        batch_target_type = []
        batch_target_mat = []
        batch_image = np.empty((batchSubset.shape[0] * (1+self.dataAug), 224, 224, 3), dtype=np.float32)
        batch_artist_weight = np.empty((batchSubset.shape[0] * (1+self.dataAug), 2), dtype=np.float32)        
        count = 0
        
        #for Id in batch_Id:
        for index, row in batchSubset.iterrows():

            # append labels
            batch_target_artist.append(row['Artist'])
            batch_artist_weight[count, 0] = row['Weight_Acc']
            batch_artist_weight[count, 1] = row['Weight_Loss']
            batch_target_year.append(row['Year_Est'])
            batch_target_type.append(stringToList(row['Type_all']))
            batch_target_mat.append(stringToList(row['Material_all']))
                       
           # load, pre-process and append image
            img = Image.open(self.dir_img + index +'.jpg')
            img = np.array(img)
            if(img.ndim != 3):
                img = np.stack((img,)*3, axis=-1)
            img_prepro = applications.resnet50.preprocess_input(img)
            batch_image[count, ...] = img_prepro
            count += 1

            # test - save img ###
#            matplotlib.image.imsave('../../test/img/' + str(self.epochRef) + '_'  + str(count) + '_' + index + '_' + row['Artist'] +'_' + str(row['Year_Est']) +'_resized.jpg', img_prepro)

            if(self.dataAug):
                # apply model specific pre-processing
                #if(MODEL_NAME == 'RESNET'):   flip_img_prepro = applications.resnet50.preprocess_input(flip_img)
                #if(MODEL_NAME == 'Xception'):    flip_img_prepro = applications.xception.preprocess_input(flip_img)
                #else: flip_img_prepro = applications.vgg16.preprocess_input(flip_img)

                # append labels                
                batch_target_artist.append(row['Artist'])
                batch_artist_weight[count, 0]  = row['Weight_Acc']
                batch_artist_weight[count, 1] = row['Weight_Loss']
                batch_target_year.append(row['Year_Est'])
                batch_target_type.append(stringToList(row['Type_all']))
                batch_target_mat.append(stringToList(row['Material_all']))

                # Basic Data Augmentation - Horizontal Flipping
                flip_img_prepro = np.fliplr(img_prepro)
                # append image
                batch_image[count, ...] = flip_img_prepro
                count += 1

                # test - save img ###
#                matplotlib.image.imsave('../../test/img/' + str(self.epochRef) + '_'  + str(count) + '_' + index + '_' + row['Artist'] +'_' +  str(row['Year_Est']) +'_resized_flipped.jpg', flip_img_prepro)          

        batch_target_artist = np.array( batch_target_artist ).reshape(-1,1)
        batch_target_artist = self.encoder_Artist.transform(batch_target_artist)
        batch_artist = np.append(batch_artist_weight, batch_target_artist, axis = 1)

        batch_target_type = self.encoder_Type.transform(batch_target_type)
        batch_target_mat = self.encoder_Mat.transform(batch_target_mat)          
        batch_target_year = np.array( batch_target_year ).reshape(-1,1)
        batch_target_year = self.year_scaler.transform(batch_target_year)
          
        return(batch_image, {'artist': batch_artist, 'year': batch_target_year,'type': batch_target_type, 'mat': batch_target_mat} )

def stringToList(text):
    types = []
    text_type = re.split(";", text)
    for typ in text_type:
        if not typ == '':
            types.append(typ)  
    return types


#################### BACKUP 
        
    def __getitem2__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation2(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)