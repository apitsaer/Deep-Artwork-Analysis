#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:03:07 2019

@author: Alexandre Pitsaer
"""
import os # used for navigating to image path
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import re
import datetime

import numpy as np
import pandas as pd
from random import shuffle
from math import ceil
from itertools import product

import collections
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, MinMaxScaler

import keras
from keras.models import Model, load_model
from keras import Input, layers, callbacks, applications, optimizers, regularizers
from keras.preprocessing import image
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

import Utils_Cust as u
import CustomMetrics as cm

def ini(MODEL_NAME, LOGS_FOLDER=''):    
    
    global nClassesArtist, nClassesType, nClassesMat, AWTable, artist_weights_matrix, artistsWeightTable, encoder_Artist, encoder_Type, encoder_Mat, year_scaler
    
    #u.get_IMG_size_statistics(DIR)
    AWTable = pd.read_csv(DIR_METADATA + META_INPUT_FILE, keep_default_na=False)
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
    
    # Using a min max scaler (0-1) for Year
    train_valid_years = pd.concat([train_set.Year_Est, valid_set.Year_Est], axis = 0)
    year_scaler = MinMaxScaler(feature_range=(0, 1))
    year_scaler.fit(np.asarray(train_valid_years).reshape(-1,1))

    # generating weights to be used by the data generators for the accuracy
    artistsFreqTable = AWTable['Artist'].value_counts(normalize=True)
    artistsWeightTable = pd.DataFrame([artistsFreqTable]).transpose()
    artistsWeightTable.rename(columns={'Artist':'Count', 'index':'Artist'}, inplace=True)
    max_count = artistsWeightTable['Count'].max()
    tot_count = artistsWeightTable['Count'].sum()    
    artistsWeightTable['Weight_Acc'] = 1 / artistsWeightTable.Count / nClassesArtist
    artistsWeightTable['Weight_Loss'] = max_count / (artistsWeightTable.Count + (max_count * W_SMOOTHING))

    # new weight calculation for the loss
    #artistsCountTable = AWTable['Artist'].value_counts(normalize=False)
    b = pd.DataFrame([artistsFreqTable]).transpose()
    b.reset_index(inplace = True)
    b.rename(columns={'Artist':'Count', 'index':'Artist'}, inplace=True)
    max_count = b['Count'].max()
    tot_count = b['Count'].sum()
    b['Weight_Acc'] = tot_count / b.Count / nClassesArtist
    b['Weight_Loss'] = max_count * tot_count / b.Count
    b['Weight_Loss_LogSmoothed'] = np.log(tot_count / b.Count)
    b['Weight_Loss_Smoothed'] = max_count / (b.Count + (max_count * W_SMOOTHING))
    b.sort_values(by=['Artist'], inplace=True)
    artists_weights = np.array(b['Weight_Loss_Smoothed'])
    
    # create matrix
    artist_weights_matrix = np.ones([nClassesArtist, nClassesArtist])
    #artist_weights_matrix = [[]]
    for c_p, c_t in product(range(nClassesArtist), range(nClassesArtist)):
        #if(c_p != c_t):
        #if(WEIGHTING != 'NO'):
        artist_weights_matrix[c_t, c_p] = artists_weights[c_t]       
    # #########################

    # writing to log file
    if(LOGS_FOLDER != ''):
        os.mkdir(LOGS_FOLDER)
        f = open(LOGS_FOLDER +  'Run_info.txt', 'a')
        msg1 = 'run description = \t{} \nMETA_INPUT_FILE = \t{} \nmodel = \t{} \nh_units = \t{} \nn_epochs = \t{} \nbatch_s = \t{} \nimg_s = \t{} \n'.format(RUN_DESCR, META_INPUT_FILE, MODEL_NAME, NUM_HIDDEN_UNITS, NUM_EPOCHS, BATCH_SIZE, IMG_SIZE)
        msg1 += 'dropout share layer =  \t{:.2f} \nweighting = \t{} \nweight smoothing= \t{:.2f} \nfocal = \t{} \n\n'.format(DROPOUT, WEIGHTING, W_SMOOTHING, FOCAL)
        msg1 += 'Found \n{} artworks \n'.format(AWTable.shape[0])
        msg1 +='{} Artists \n{} Types \n{} Mats \n'.format(nClassesArtist, nClassesType, nClassesMat)
        msg1 +='Crea. Year from {:.0f} \t to {:.0f} \t Average = {:.0f} \n'.format(AWTable['Year_Est'].min(), AWTable['Year_Est'].max(), AWTable['Year_Est'].mean())
        msg1 +='Size train = {} \t valid = {} \t test = {} \n\n'.format(train_set.shape[0], valid_set.shape[0], test_set.shape[0])
        msg1 +='Min weight acc = {:2f} / Min weight acc = {:2f}'.format(artistsWeightTable['Weight_Acc'].min(), artistsWeightTable['Weight_Acc'].max())
        msg1 +='Min weight loss = {:2f} / Min weight loss = {:2f} \n\n'.format(artistsWeightTable['Weight_Loss'].min(), artistsWeightTable['Weight_Loss'].max())
        print(msg1)    
        f.write(msg1)
        f.close()

    return (train_set, valid_set, test_set)

def learnModelMulti(MODEL_FOLDER, MODEL_NAME, LOGS_FOLDER, train_set, valid_set):
    
    if(MODEL_NAME == 'RESNET'):
        K.set_learning_phase(0) # set model to inference / test mode manually (required for BN layers)
        base_model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        #x = layers.Dropout(0.5)(x)
        #base_model.summary()
    elif(MODEL_NAME == 'Xception'):
        K.set_learning_phase(0) # set model to inference / test mode manually (required for BN layers)
        base_model = applications.Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        #ox = layers.Dropout(0.7)(x)
    else:
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model.output
        x = layers.Flatten()(x)     # flatten 3D output to 1D
    
    print('***** Base model {} loaded:'.format(MODEL_NAME))

    #n_layers_base = len(base_model.layers)
    
    if(MODEL_NAME == 'Xception' or MODEL_NAME == 'RESNET'):
        K.set_learning_phase(1) # set model to training mode manually (required for BN layers)

    MLT_shared_repr = layers.Dense(NUM_HIDDEN_UNITS, name='shared_repr')(x)        
    MLT_shared_repr = layers.BatchNormalization()(MLT_shared_repr)
    MLT_shared_repr = layers.Activation('relu')(MLT_shared_repr)

#    MLT_shared_repr = layers.Dense(NUM_HIDDEN_UNITS, activation = 'relu', name='shared_repr')(x)        
#    MLT_shared_repr = layers.Activation('relu')(MLT_shared_repr)

    drop_out_layer = layers.Dropout(DROPOUT)(MLT_shared_repr)
    artist_prediction = layers.Dense(nClassesArtist, activation='softmax', name='artist', kernel_regularizer=regularizers.l2(0.001))(drop_out_layer)
    year_prediction = layers.Dense(1, name='year')(drop_out_layer) # regression hence no activation function
    type_prediction = layers.Dense(nClassesType, activation='sigmoid', name='type')(drop_out_layer)
    mat_prediction = layers.Dense(nClassesMat, activation='sigmoid', name='mat')(drop_out_layer)

    global custom_model
    custom_model = Model(base_model.input,[artist_prediction, year_prediction, type_prediction, mat_prediction])
    
    #print("***** Full model")
    #custom_model.summary()
    
    print('# trainable weights '
          'before freezing the conv base:', len(custom_model.trainable_weights))
    
    for layer in base_model.layers:
        layer.trainable = False

    print('# trainable weights '
          'after freezing the conv base:', len(custom_model.trainable_weights))
    
    artist_loss_weight = 1
    year_loss_weight = 0 #0.1 #0.05
    type_loss_weight = 1
    mat_loss_weight = 1
    
    f = open(LOGS_FOLDER + 'Run_info.txt', 'a')
    msg = '\nartist_loss_weight = {:.2f}\n'.format(artist_loss_weight)
    msg += 'year_loss_weight = {:.2f}\n'.format(year_loss_weight)
    msg += 'type_loss_weight = {:.2f}\n'.format(type_loss_weight)
    msg += 'mat_loss_weight = {:.2f}\n'.format(mat_loss_weight)
    print(msg)    
    f.write(msg)
    f.close()
    
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=2e-3, decay=0.9)
    #custom_model.compile(optimizer='rmsprop',
    #sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True) 
    mae_tol = cm.mae_tol_param(0.15)
    #loss_artits_w = cm.categorical_crossentropy_w_wrap(artist_weights_matrix)
    
    if(FOCAL and WEIGHTING):
        artistLoss = cm.w_categorical_focal_loss(alpha=1, gamma=2)
    elif (FOCAL and not WEIGHTING):
        artistLoss = cm.categorical_focal_loss(alpha=1, gamma=2)
    elif (not FOCAL and WEIGHTING):
        artistLoss = lambda y_true, y_pred: cm.w_categorical_crossentropy(y_true, y_pred, weights=artist_weights_matrix)
    elif (not FOCAL and not WEIGHTING):
        artistLoss = cm.categorical_crossentropy_abs

    custom_model.compile(optimizer='adam',
                  loss={'artist': artistLoss, 'year': 'mae', 'type': 'binary_crossentropy', 'mat': 'binary_crossentropy'},
                  loss_weights={'artist': artist_loss_weight, 'year': year_loss_weight, 'type': type_loss_weight, 'mat': mat_loss_weight},
                  metrics={'artist': [cm.accuracy_abs, cm.accuracy_w] , 'year': ['mae', mae_tol] , 'type': cm.precision, 'mat': cm.precision})
    
    train_gen = data_generator(MODEL_NAME, train_set, dataAugm=True)
    valid_gen = data_generator(MODEL_NAME, valid_set, dataAugm=False)
    
    #callbacks = [keras.callbacks.TensorBoard(log_dir='../logs',histogram_freq=1]    
    #csv_logger = callbacks.CSVLogger(LOGS_FOLDER + MODEL_NAME +'.log')

    callbacks_list = [callbacks.CSVLogger(LOGS_FOLDER + MODEL_NAME +'.log'),
                         #callbacks.EarlyStopping(monitor='val_artist_accuracy_w',patience=4),
                         callbacks.ModelCheckpoint(filepath = MODEL_FOLDER + MODEL_NAME + '.h5',monitor='val_artist_accuracy_w',save_best_only=True)]

    history = custom_model.fit_generator(train_gen, 
                        validation_data = valid_gen,  
                        validation_steps = ceil(valid_set.shape[0] / BATCH_SIZE),
                        steps_per_epoch = ceil(2 * train_set.shape[0] / BATCH_SIZE),
                        epochs = NUM_EPOCHS, verbose = 1,
                        callbacks = callbacks_list)

    print('***** Training logs saved as ' + LOGS_FOLDER + MODEL_NAME +'.log')

    #custom_model.save(LOGS_FOLDER + 'artistsRD.h5')
    print('***** Model saved as artistsRD.h5')
 
# evaluate on test set
def testModelMulti(MODEL_FOLDER, MODEL_NAME, test_set, LOGS_FOLDER=''):
    if(MODEL_NAME == 'Xception' or MODEL_NAME == 'RESNET'):
        K.set_learning_phase(0) # set model to inference / test mode manually (required for BN layers)
    #model = custom_model
    model = load_model(MODEL_FOLDER + MODEL_NAME + '.h5', custom_objects={'precision': cm.precision, 'accuracy_abs': cm.accuracy_abs, 'accuracy_w': cm.accuracy_w, 'categorical_focal_loss_fixed': cm.categorical_focal_loss(alpha=.25, gamma=2), 'w_categorical_focal_loss_fixed': cm.w_categorical_focal_loss(alpha=.25, gamma=2), 'mae_tol': cm.mae_tol_param(0.15)})
    test_gen = data_generator(MODEL_NAME, test_set, dataAugm=False)    

    loss, artist_loss, year_loss, type_loss, mat_loss, artist_accuracy_abs, artist_accuracy_w, year_mean_absolute_error, year_mae_tol, type_precision, mat_precision = model.evaluate_generator(test_gen, 
                                                                                  steps = ceil(test_set.shape[0]/BATCH_SIZE))

    msg = '******\n'
    msg += 'total_loss = {:.2f}\n'.format(loss)
    msg += 'test_artist_loss = {:.2f}\n'.format(artist_loss)
    msg += 'year_loss = {:.2f}\n'.format(year_loss)
    msg += 'type_loss = {:.2f}\n'.format(type_loss)
    msg += 'mat_loss = {:.2f}\n'.format(mat_loss)
    msg += '******\n'
    msg += 'test_artist_acc_abs = {:.2%}\n' .format(artist_accuracy_abs)
    msg += 'test_artist_acc_w = {:.2%}\n' .format(artist_accuracy_w)
    msg += 'test_year_mean_absolute_error = {:.2}\n'.format(year_mean_absolute_error)
    msg += 'test_type_precision = {:.2%}\n'.format(type_precision)
    msg += 'test_mat_precision = {:.2%}\n'.format(mat_precision)
    print(msg)    

    if(LOGS_FOLDER == ''):
        f = open(LOGS_FOLDER +'Run_info.txt', 'a')
    else:
        f = open(MODEL_FOLDER +'test_info.txt', 'a')        
    f.write(msg)
    f.close()
    
# show some prediction
def predictModelMulti(MODEL_FOLDER, MODEL_NAME, test_set, LOGS_FOLDER=''):
    if(MODEL_NAME == 'Xception' or MODEL_NAME == 'RESNET'):
        K.set_learning_phase(0) # set model to inference / test mode manually (required for BN layers)
    #model = custom_model
    model = load_model(MODEL_FOLDER + MODEL_NAME + '.h5', custom_objects={'precision': cm.precision, 'accuracy_abs': cm.accuracy_abs, 'accuracy_w': cm.accuracy_w, 'categorical_focal_loss_fixed': cm.categorical_focal_loss(alpha=.25, gamma=2), 'w_categorical_focal_loss_fixed': cm.w_categorical_focal_loss(alpha=.25, gamma=2), 'mae_tol': cm.mae_tol_param(0.15)})
    test_gen = data_generator(MODEL_NAME, test_set, dataAugm=False)    
    predict_gen = model.predict_generator(test_gen, steps = ceil(test_set.shape[0]/BATCH_SIZE))
    predicted_years_list = year_scaler.inverse_transform(predict_gen[1])
    
    predTable = pd.DataFrame(columns=['Id', 'Artist_Pred','Artist_Act','Artist_OK','Year_Pred', 'Year_Act', 'Year_Err', 'Type_Act', 'Material_Act', 'Title', 'Description'])
    
    for index in range(len(predict_gen[0])):
        imgId = test_set.index[index]
        actual_year = test_set.iloc[index].Year_Est
        #predicted_year = round(predict_gen[1][index][0])
        predicted_year = predicted_years_list[index][0]
        err_year = round(abs(actual_year - predicted_year))
        actual_artist = test_set.iloc[index].Artist
        predicted_artist = encoder_Artist.inverse_transform([predict_gen[0][index]])[0][0]
        acc_artist = (actual_artist == predicted_artist)
#            artist_weight  = artistsWeightTable.loc[actual_artist].Weight
        actual_type = AWTable.loc[imgId].Type_all
        actual_mat = AWTable.loc[imgId].Material_all
        title = AWTable.loc[imgId].Title_all
        description = AWTable.loc[imgId].Description_all       
        predTable.loc[index] = [imgId, predicted_artist, actual_artist, acc_artist, predicted_year, actual_year, err_year, actual_type, actual_mat, title, description]
   
    if(LOGS_FOLDER == ''):
        LOGS_FOLDER = MODEL_FOLDER
    predTable.to_csv(LOGS_FOLDER + 'Predictions.csv', index=False, encoding='utf8')
    print('***** Predictions on test set saved in ' + LOGS_FOLDER + 'Predictions.csv')
    

################# MAIN
    
def MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 512,  n_epochs = 40, batch_s = 20, img_s = 200, dropout = 0.25, focal = True, weighting = True, w_smooth = 0.2, descr = 'No info'):

    global WEIGHTING, W_SMOOTHING, FOCAL, DROPOUT, META_INPUT_FILE, DIR_IMG, DIR_METADATA, NUM_EPOCHS, BATCH_SIZE, IMG_SIZE, NUM_HIDDEN_UNITS, RUN_DESCR

    DIR_IMG = '../../data/1_original/small/' #must change to previous
    DIR_METADATA = '../../data/2_meta_files/'
    DIR_LOGS_BASE = '../../logs/' 
    MODEL_FOLDER_BASE = '../../models/'
    NUM_EPOCHS = n_epochs
    BATCH_SIZE = batch_s
    IMG_SIZE = img_s
    MODEL_NAME = model
    NUM_HIDDEN_UNITS = h_units
    RUN_DESCR = descr
    DROPOUT = dropout
    FOCAL = focal
    WEIGHTING = weighting
    W_SMOOTHING = w_smooth
    META_INPUT_FILE = input_file
    
    loss_descr = '_FOCAL' + str(FOCAL) + '_WEIGHTING' + str(WEIGHTING) + '_WS' + str(W_SMOOTHING)
    
    datetime_object = datetime.datetime.now()
    LOGS_FOLDER = DIR_LOGS_BASE  + str(datetime_object.day) + '-' + str(datetime_object.month) + '_' + str(datetime_object.hour) + '.' + str(datetime_object.minute) + '_' + META_INPUT_FILE + '_' + MODEL_NAME +'_Un' + str(h_units) + '_EPOCH' +  str(NUM_EPOCHS) + '_BS' + str(BATCH_SIZE) + '_IS'+ str(IMG_SIZE) + '/'
    MODEL_FOLDER = MODEL_FOLDER_BASE +'_' + META_INPUT_FILE + '_' + MODEL_NAME + '_Un' +  str(h_units) + '_EPOCH' +  str(NUM_EPOCHS) + '_BS' + str(BATCH_SIZE) +  loss_descr + '/'
    
    if os.path.exists(MODEL_FOLDER):
        shutil.rmtree(MODEL_FOLDER)
    os.mkdir(MODEL_FOLDER)
        

    print("*****    1. Generating data sets and initializing variables")
    (train_set, valid_set, test_set) = ini(MODEL_NAME, LOGS_FOLDER)
    
    debugDataGen(train_set, valid_set, test_set)

    print("*****    2. Learning")
    learnModelMulti(MODEL_FOLDER, MODEL_NAME, LOGS_FOLDER, train_set, valid_set)
    u.makePlots(LOGS_FOLDER, MODEL_NAME)
    
    print("*****    3. Evaluating on test set")
    testModelMulti(MODEL_FOLDER, MODEL_NAME, test_set, LOGS_FOLDER)
    
    print("*****    4. Predicting on test set")
    predictModelMulti(MODEL_FOLDER, MODEL_NAME, test_set, LOGS_FOLDER)    
    
def MLT_test(input_file = 'AWTableTOP20.csv', model_foler = '../../models/', model = 'RESNET', w_smooth = 0.2, batch_size = 50):

    global WEIGHTING, W_SMOOTHING, FOCAL, DROPOUT, LOGS_FOLDER, META_INPUT_FILE, DIR_IMG, DIR_METADATA, NUM_EPOCHS, BATCH_SIZE, IMG_SIZE, NUM_HIDDEN_UNITS, RUN_DESCR

    DIR_IMG = '../../data/1_original/small/' #must change to previous
    DIR_METADATA = '../../data/2_meta_files/'
    MODEL_FOLDER = model_foler
    MODEL_NAME = model
    META_INPUT_FILE = input_file
    W_SMOOTHING = w_smooth
    BATCH_SIZE = batch_size

    print("*****    1. Generating data sets and initializing variables")
    (train_set, valid_set, test_set) = ini(MODEL_NAME)

    print("*****    2. Evaluating on test set")
    testModelMulti(MODEL_FOLDER, MODEL_NAME, test_set)
    
    print("*****    3. Predicting on test set")
    predictModelMulti(MODEL_FOLDER, MODEL_NAME, test_set)    

# Custom data generator to provide batch of labelized data
def data_generator(MODEL_NAME, data_set, dataAugm=False):
    
  
    from timeit import default_timer as timer
    #import cv2
       
    iBatch = 0
    steps_per_epoch = ceil(data_set.shape[0] / BATCH_SIZE)
    
    start_batch = timer()

     
    while True:
 
        batch_Id = data_set.iloc[iBatch * BATCH_SIZE : (iBatch+1) * BATCH_SIZE].index

        #batch_image = []
        batch_image = np.empty((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
        batch_target_artist = [] 
        batch_artist_weight_acc = []
        batch_artist_weight_loss = []
        batch_target_year = []
        batch_target_type = []
        batch_target_mat = []
        
        count = 0

        for Id in batch_Id:

            artist = AWTable.loc[Id].Artist
            artist_weight_acc  = artistsWeightTable.loc[artist].Weight_Acc
            artist_weight_loss  = artistsWeightTable.loc[artist].Weight_Loss
            year = AWTable.loc[Id].Year_Est
            types = AWTable.loc[Id].Type_all
            mats = AWTable.loc[Id].Material_all
            
            batch_target_artist.append(artist)
            batch_artist_weight_acc.append(artist_weight_acc)
            batch_artist_weight_loss.append(artist_weight_loss)
            batch_target_year.append(year)
            batch_target_type.append(stringToList(types))
            batch_target_mat.append(stringToList(mats))
            path = os.path.join(DIR_IMG, Id +".jpg")
             
            img = image.load_img(path)
            img = image.img_to_array(img)
            
            # apply model specific pre-processing
            if(MODEL_NAME == 'RESNET'):    
                img_prepro = applications.resnet50.preprocess_input(img)
            elif(MODEL_NAME == 'Xception'):
                img_prepro = applications.xception.preprocess_input(img)
            else:
                img_prepro = applications.vgg16.preprocess_input(img)

            batch_image[count, ...] = img_prepro
            count += 1
            
            if(dataAugm):
                # Basic Data Augmentation - Horizontal Flipping
                flip_img_prepro = np.fliplr(img_prepro)
                # apply model specific pre-processing
                #if(MODEL_NAME == 'RESNET'):   flip_img_prepro = applications.resnet50.preprocess_input(flip_img)
                #if(MODEL_NAME == 'Xception'):    flip_img_prepro = applications.xception.preprocess_input(flip_img)
                #else: flip_img_prepro = applications.vgg16.preprocess_input(flip_img)
                
                batch_image.append(np.array(flip_img_prepro))
                  
                # test - save img
                #matplotlib.image.imsave('../../test/img/' + Id + '_' + artist +'_' + str(year) +'_resized.jpg', img_prepro)
                # test - save img
                #matplotlib.image.imsave('../../test/img/' + Id + '_' + artist +'_' +  str(year) +'_resized_flipped.jpg', flip_img_prepro)
                
                batch_target_artist.append(artist)
                batch_artist_weight_acc.append(artist_weight_acc)
                batch_artist_weight_loss.append(artist_weight_loss)
                batch_target_year.append(year)
                batch_target_type.append(stringToList(types))
                batch_target_mat.append(stringToList(mats))

        #batch_image = np.array( batch_image) # / 255 # remove if using model pre-processing
          
        batch_target_artist = np.array( batch_target_artist ).reshape(-1,1)
        batch_target_artist = encoder_Artist.transform(batch_target_artist)
        batch_artist_weight_acc = np.array( batch_artist_weight_acc).reshape(-1,1)
        batch_artist_weight_loss = np.array( batch_artist_weight_loss).reshape(-1,1)
        # batch_artist is concatenation of artist weight (first column) and artist values one hot encoded (remaining columns)
        batch_artist = np.append(batch_artist_weight_acc, batch_artist_weight_loss, axis = 1)
        batch_artist = np.append(batch_artist, batch_target_artist, axis = 1)
        
        batch_target_type = encoder_Type.transform(batch_target_type)
        batch_target_mat = encoder_Mat.transform(batch_target_mat)          
        batch_target_year = np.array( batch_target_year ).reshape(-1,1)
        batch_target_year = year_scaler.transform(batch_target_year)
          
        endtransorm = timer()
        print('batch in sec = {:2f} \n\n'.format(endtransorm - start_batch)) # Time in seconds, e.g. 5.38091952400282
        
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
    
    
def debugDataGen(train_set, valid_set, test_set):
        # ======== CODE to DEBUG the generator =======================================
    train_gen = data_generator('RESNET',  train_set)
    valid_gen = data_generator('RESNET', valid_set)
    test_gen = data_generator('RESNET', test_set)
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


MLT_learn_test(input_file = 'AWTableSetA.csv', model = 'RESNET', h_units = 1500,  n_epochs = 40, batch_s = 10, img_s = 224, dropout = 0.25, focal = False, weighting = False, w_smooth = 0.2,  descr = 'rmsprop  no weights, 1 dropout 0.25 + no L2 reg artist + no year')
#MLT_test(input_file = 'AWTableTOP100.csv', model_foler = '../../models/', model = 'RESNET')

# To DO 
# stemming to material
# dictionary: add last name of artist
# run on full dataset
# 2. Use Keras data augmentation
# try online data augmentation
# implement step decay

#inverted = encoder_Artist.inverse_transform([dicto.get('artist')[1]])[0][0]
#encoder_Artist.inverse_transform([[1,0,0,0]])[0][0]

#   Code to use Keras built in data augm
#    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
#                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
#                             horizontal_flip=True, fill_mode="nearest")
    

 
