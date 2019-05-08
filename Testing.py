#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:09:44 2019

@author: admin
"""
import os # used for navigating to image path
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re
import datetime

import numpy as np
import pandas as pd
from random import shuffle
from math import ceil

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, MinMaxScaler

import keras
from keras.models import Model, load_model
from keras import Input, layers, callbacks, applications, optimizers
from keras.preprocessing import image
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

import Utils as u
import CustomMetrics as cm
from MLT_W import data_generator
    
def MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 512,  n_epochs = 40, batch_s = 20, img_s = 200, descr = 'No info'):

    global LOGS_FOLDER, DIR_IMG, AWTable, artistsWeightTable, encoder_Artist, encoder_Type, encoder_Mat, nClassesArtist, nClassesType, nClassesMat, train_set, valid_set, test_set, NUM_EPOCHS, BATCH_SIZE, IMG_SIZE, MODEL_NAME, NUM_HIDDEN_UNITS, RUN_DESCR

    #DIR_IMG = '../../data/1_original/img/'
    DIR_IMG = '../../data/TOP100/original/' #must change to previous
    DIR_METADATA = '../../data/2_meta_files/'
    DIR_LOGS_BASE = '../../logs/' 
    NUM_EPOCHS = n_epochs
    BATCH_SIZE = batch_s
    IMG_SIZE = img_s
    MODEL_NAME = model
    NUM_HIDDEN_UNITS = h_units
    RUN_DESCR = descr
    
    print("*****    1. Generating data sets")
    AWTable = pd.read_csv(DIR_METADATA + input_file, keep_default_na=False)
    AWTable.set_index("Id", inplace=True)
    AWTable.columns = AWTable.columns.str.strip() #remove leading and trailing white space if any
    
    (all_Types, all_Mats) = u.getAllTypeMat(AWTable)
    nClassesArtist = AWTable['Artist'].nunique()
    nClassesType = len(all_Types)
    nClassesMat = len(all_Mats)
        
    # generating training, validation and test data sets with stratified shuffle split
    # using Artists as key attribute for the stratification        
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_valid_index in split.split(AWTable, AWTable.Artist):
        train_set = AWTable.iloc[train_index]
        test_valid_set = AWTable.iloc[test_valid_index]
    #Also build a validation set by splitting the previously generated test set
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for test_index, valid_index in split2.split(test_valid_set, test_valid_set.Artist):
        test_set = test_valid_set.iloc[test_index]
        valid_set = test_valid_set.iloc[valid_index]

    msg2 = 'Found \n{} artworks \n'.format(AWTable.shape[0])
    msg2 +='{} Artists \n{} Types \n{} Mats \n'.format(nClassesArtist, nClassesType, nClassesMat)
    msg2 +='Crea. Year from {:.0f} \t to {:.0f} \t Average = {:.0f} \n'.format(AWTable['Year_Est'].min(), AWTable['Year_Est'].max(), AWTable['Year_Est'].mean())
    msg2 +='Size train = {} \t valid = {} \t test = {} \n\n'.format(train_set.shape[0], valid_set.shape[0], test_set.shape[0])
    print(msg2)    
    
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
    all_Artists = AWTable.Artist.unique().reshape(-1,1)
    encoder_Artist.fit(all_Artists)
    
    # generating weights to be used by the data generators for the custom loss and accuracy
    artistsFreqTable = AWTable['Artist'].value_counts(normalize=True)
    artistsWeightTable = pd.DataFrame([artistsFreqTable]).transpose()
    artistsWeightTable['Weight'] = 1 / artistsWeightTable.Artist / nClassesArtist
    
    debugDataGen()

    
def debugDataGen():
        # ======== CODE to DEBUG the generator =======================================
    train_gen = data_generator(train_set, AWTable, DIR_IMG, BATCH_SIZE, IMG_SIZE)
    valid_gen = data_generator(valid_set, AWTable, DIR_IMG, BATCH_SIZE, IMG_SIZE)
    test_gen = data_generator(test_set, AWTable, DIR_IMG, BATCH_SIZE, IMG_SIZE)
    steps_per_epoch = ceil(valid_set.shape[0] / BATCH_SIZE)
    (batch_image, dicto) = next(train_gen)
    print('Artist weight')
    print(dicto.get('artist')[:,0])
    print('Artist attribu')
    print(dicto.get('artist')[:,1:])    
    (batch_image, dicto) = next(train_gen)
    print('Artist weight')
    print(dicto.get('artist')[:,0])
    print('Artist attribu')
    print(dicto.get('artist')[:,1:])    

def debugMetrics():
    # =============================================================================
    # ======== CODE to DEBUG the custon metrics and losses ============================
    from keras.activations import softmax
    from keras.objectives import categorical_crossentropy
    import keras.backend as K
    from keras.metrics import categorical_accuracy, categorical_crossentropy    
    
    # testing weighted acc for Artist
    y_true = dicto.get('artist')
    y_pred = np.copy(y_true[:,1:])
    y_pred[0] = [0,0,0,1, 0]
    y_pred[1] = [0,0,1,0, 0]
    acc_abs = cm.accuracy_abs(y_true,y_pred).eval(session=K.get_session())
    #np.testing.assert_almost_equal(acc_weighted,acc_abs)    
    weights = y_true[:,0]
    y_true = y_true [:,1:]
    acc_weighted = cm.accuracy_w(y_true,y_pred).eval(session=K.get_session())    
    
    # testing mae with tol for Year
    y_year_true = dicto.get('year')
    tol = 50
    y_year_delta = np.random.random_integers(tol - 10, tol + 20, size=(y_year_true.shape[0],1))
    y_year_pred = y_year_true - y_year_delta
    
    true_mae_tol = 0
    for i in range(0, y_year_delta.shape[0]):
        if(y_year_delta[i,0] > tol):
            true_mae_tol += (y_year_delta[i,0] - tol)
    true_mae_tol = true_mae_tol / y_year_delta.shape[0]
    
    abs_diff = K.abs(y_year_true - y_year_pred)
    abs_diff_tol = K.maximum(abs_diff - tol, 0)
    mae_tol1 = K.mean(abs_diff_tol)
    mae_tol1 = mae_tol1.eval(session=K.get_session())
    np.testing.assert_almost_equal(true_mae_tol,mae_tol1)
    print('test 1 mae tol OK')
    
    mae_tol2 = cm.mae_tol_param(tol)(y_year_true,y_year_pred)
    mae_tol2 = mae_tol2.eval(session=K.get_session())
    np.testing.assert_almost_equal(true_mae_tol,mae_tol2)
    print('test 2 mae tol OK')
    
MLT_learn_test(input_file = 'AWTableSetA.csv', model = 'RESNET', h_units = 512,  n_epochs = 1, batch_s = 10, img_s = 200)
