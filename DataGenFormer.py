#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:17:07 2019

@author: admin
"""
from PIL import Image
import numpy as np
import re
from math import ceil
from keras import applications



# Custom data generator to provide batch of labelized data
def data_generator(MODEL_NAME, data_set, dir_img, encoder_Artist, encoder_Type, encoder_Mat,
                 year_scaler, batch_size, dataAugm=False):
       
    iBatch = 0
    # defining the steps per epoch dependng on data augmentation (times 2)
    steps_per_epoch = ceil( data_set.shape[0] / batch_size)
    
    while True:
 
        batchSubset = data_set.iloc[iBatch * batch_size : min((iBatch+1) * batch_size, data_set.shape[0]), :]
        
        # shuffe batch subset     
        #data_set.iloc[29 * batch_size : min((29+1) * batch_size, data_set.shape[0]), :].shape[0]
        
        batch_target_artist = [] 
        batch_target_year = []
        batch_target_type = []
        batch_target_mat = []
        batch_image = np.empty((batchSubset.shape[0] * (1+dataAugm), 224, 224, 3), dtype=np.float32)
        batch_artist_weight = np.empty((batchSubset.shape[0] * (1+dataAugm), 2), dtype=np.float32)        
        count = 0

        #for Id in batch_Id:
        for index, row in batchSubset.iterrows():

            batch_target_artist.append(row['Artist'])
            batch_artist_weight[count, 0] = row['Weight_Acc']
            batch_artist_weight[count, 1] = row['Weight_Loss']
            batch_target_year.append(row['Year_Est'])
            batch_target_type.append(stringToList(row['Type_all']))
            batch_target_mat.append(stringToList(row['Material_all']))
                       
           # load the image
            img = Image.open(dir_img + index +'.jpg')
            img = np.array(img)
            #print('img shape = {} '.format(img.ndim))
            if(img.ndim != 3):
                img = np.stack((img,)*3, axis=-1)
            img_prepro = applications.resnet50.preprocess_input(img)
#            elif(MODEL_NAME == 'Xception'):
#                img_prepro = applications.xception.preprocess_input(img)
#            else:
#                img_prepro = applications.vgg16.preprocess_input(img)

            batch_image[count, ...] = img_prepro
            count += 1

            if(dataAugm):
                # Basic Data Augmentation - Horizontal Flipping
                flip_img_prepro = np.fliplr(img_prepro)
                # apply model specific pre-processing
                #if(MODEL_NAME == 'RESNET'):   flip_img_prepro = applications.resnet50.preprocess_input(flip_img)
                #if(MODEL_NAME == 'Xception'):    flip_img_prepro = applications.xception.preprocess_input(flip_img)
                #else: flip_img_prepro = applications.vgg16.preprocess_input(flip_img)
                
                batch_target_artist.append(row['Artist'])
                batch_artist_weight[count, 0] = row['Weight_Acc']
                batch_artist_weight[count, 1] = row['Weight_Loss']
                batch_target_year.append(row['Year_Est'])
                batch_target_type.append(stringToList(row['Type_all']))
                batch_target_mat.append(stringToList(row['Material_all']))

                batch_image[count, ...] = flip_img_prepro
                count += 1

#                # test - save img
#                matplotlib.image.imsave('../../test/img/' + index + '_' + row['Artist'] +'_' + str(row['Year_Est']) +'_resized.jpg', img_prepro)
#                # test - save img
#                matplotlib.image.imsave('../../test/img/' + index + '_' + row['Artist'] +'_' +  str(row['Year_Est']) +'_resized_flipped.jpg', flip_img_prepro)

        batch_target_artist = np.array( batch_target_artist ).reshape(-1,1)
        batch_target_artist = encoder_Artist.transform(batch_target_artist)
        batch_artist = np.append(batch_artist_weight, batch_target_artist, axis = 1)

        batch_target_type = encoder_Type.transform(batch_target_type)
        batch_target_mat = encoder_Mat.transform(batch_target_mat)          
        batch_target_year = np.array( batch_target_year ).reshape(-1,1)
        batch_target_year = year_scaler.transform(batch_target_year)
          
        yield(batch_image, {'artist': batch_artist, 'year': batch_target_year,'type': batch_target_type, 'mat': batch_target_mat} )
        iBatch += 1
        if(iBatch == steps_per_epoch): iBatch = 0 
                
def stringToList(text):
    types = []
    text_type = re.split(";", text)
    for typ in text_type:
        if not typ == '':
            types.append(typ)  
    return types
