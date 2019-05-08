#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:03:07 2019

@author: Alexandre Pitsaer
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

def learnModelMulti():
    
    if(MODEL_NAME == 'RESNET'):
        K.set_learning_phase(0) # set model to inference / test mode manually (required for BN layers)
        base_model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        #x = layers.Dropout(0.5)(x)
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
    
    # fully connected share representation    
    x = layers.Dense(NUM_HIDDEN_UNITS, activation='relu', name='shared_repr_dense_1')(x)        
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(NUM_HIDDEN_UNITS, activation='relu', name='shared_repr_dense_2')(x)        
    x = layers.Dropout(0.5)(x)
    
    y_a = layers.Dense(NUM_HIDDEN_UNITS, activation='relu', name='dense_a')(x)        
    y_a = layers.Dropout(0.5)(y_a)
    
    y_y = layers.Dense(NUM_HIDDEN_UNITS, activation='relu', name='dense_y')(x)        
    y_y = layers.Dropout(0.5)(y_y)

    y_t = layers.Dense(NUM_HIDDEN_UNITS, activation='relu', name='dense_t')(x)        
    y_t = layers.Dropout(0.5)(y_t)

    y_m = layers.Dense(NUM_HIDDEN_UNITS, activation='relu', name='dense_m')(x)        
    y_m = layers.Dropout(0.5)(y_m)
    
    artist_prediction = layers.Dense(nClassesArtist, activation='softmax', name='artist')(y_a)
    year_prediction = layers.Dense(1, name='year')(y_y) # regression hence no activation function
    type_prediction = layers.Dense(nClassesType, activation='sigmoid', name='type')(y_t)
    mat_prediction = layers.Dense(nClassesMat, activation='sigmoid', name='mat')(y_m)

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
    
    artist_loss_weight = 2
    year_loss_weight = 0 #0.05
    type_loss_weight = 0
    mat_loss_weight = 0
    
    f = open(LOGS_FOLDER + MODEL_NAME + '_info.txt', 'a')
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
    loss_mae_tol = cm.mae_tol_param(50)

    custom_model.compile(optimizer='adam',
                  loss = {'artist': 'categorical_crossentropy', 'year': 'mae', 'type': 'binary_crossentropy', 'mat': 'binary_crossentropy'},
                  loss_weights = {'artist': artist_loss_weight, 'year': year_loss_weight, 'type': type_loss_weight, 'mat': mat_loss_weight},
                  metrics = {'artist': 'accuracy' , 'year': ['mae', loss_mae_tol] , 'type': cm.precision, 'mat': cm.precision},
                  weighted_metrics = {'artist': 'categorical_accuracy', 'year': 'mae'},
                  sample_weight_mode ={'artist': None,  'year': None , 'type': None, 'mat': None})
    
    train_gen = data_generator(train_set, Train=True)
    valid_gen = data_generator(valid_set, Train=False)
    
    #callbacks = [keras.callbacks.TensorBoard(log_dir='../logs',histogram_freq=1]    
    
    csv_logger = callbacks.CSVLogger(LOGS_FOLDER + MODEL_NAME +'.log')

    history = custom_model.fit_generator(train_gen, 
                        validation_data = valid_gen,  
                        validation_steps = ceil(valid_set.shape[0] / BATCH_SIZE),
                        steps_per_epoch = ceil(2 * train_set.shape[0] / BATCH_SIZE),
                        epochs = NUM_EPOCHS, verbose = 1,
                        #class_weight = {'artist': dic_artist_weights},
                        callbacks=[csv_logger])

    print('***** Training logs saved as ' + LOGS_FOLDER + MODEL_NAME +'.log')

    custom_model.save(MODEL_FOLDER + 'artistsRD_W.h5')
    print('***** Model saved as artistsRD.h5')
 
# evaluate on test set
def testModelMulti():
    if(MODEL_NAME == 'Xception' or MODEL_NAME == 'RESNET'):
        K.set_learning_phase(0) # set model to inference / test mode manually (required for BN layers)
    model = custom_model
    #model = load_model(LOGS_FOLDER + 'artistsRD.h5', custom_objects={'precision': cm.precision})
    test_gen = data_generator(test_set, Train=False)    

    loss, artist_loss, year_loss, type_loss, mat_loss, artist_acc, artist_weighted_categorical_accuracy, year_mean_absolute_error, year_mae_tol, year_weighted_mean_absolute_error, type_precision, mat_precision = model.evaluate_generator(test_gen, 
                                                                                  steps = ceil(test_set.shape[0]/BATCH_SIZE))

    f = open(LOGS_FOLDER + MODEL_NAME + '_info.txt', 'a')
    msg = '******\n'
    msg += 'total_loss = {:.2f}\n'.format(loss)
    msg += 'test_artist_loss = {:.2f}\n'.format(artist_loss)
    msg += 'year_loss = {:.2f}\n'.format(year_loss)
    msg += 'type_loss = {:.2f}\n'.format(type_loss)
    msg += 'mat_loss = {:.2f}\n'.format(mat_loss)
    msg += '******\n'
    msg += 'test_artist_acc_abs = {:.2%}\n' .format(artist_acc)
    msg += 'test_artist_acc_w = {:.2%}\n' .format(artist_weighted_categorical_accuracy)
    msg += 'test_year_mean_absolute_error = {:.2}\n'.format(year_mean_absolute_error)
    msg += 'test_year_mean_absolute_error with tol = {:.2}\n'.format(year_mae_tol)
    msg += 'test_type_precision = {:.2%}\n'.format(type_precision)
    msg += 'test_mat_precision = {:.2%}\n'.format(mat_precision)
    print(msg)    
    f.write(msg)
    f.close()
    
# show some prediction
def predictModelMulti():
    if(MODEL_NAME == 'Xception' or MODEL_NAME == 'RESNET'):
        K.set_learning_phase(0) # set model to inference / test mode manually (required for BN layers)
    model = custom_model
    #model = load_model(LOGS_FOLDER + 'artistsRD.h5', custom_objects={'precision': cm.precision})
    test_gen = data_generator(test_set, Train=False)    
    predict_gen = model.predict_generator(test_gen, steps = ceil(test_set.shape[0]/BATCH_SIZE))
    predicted_years_list = year_scaler.inverse_transform(predict_gen[1])
    
    predTable = pd.DataFrame(columns=['Id', 'Artist_Pred','Artist_Act','Artist_OK', 'Artist_Weight','Year_Pred', 'Year_Act', 'Year_Err', 'Type_Act', 'Material_Act', 'Title', 'Description'])
    
    for index in range(len(predict_gen[0])):
        imgId = test_set.index[index]
        actual_year = test_set.iloc[index].Year_Est
        #predicted_year = round(predict_gen[1][index][0])
        predicted_year = predicted_years_list[index][0]
        err_year = round(abs(actual_year - predicted_year))
        actual_artist = test_set.iloc[index].Artist
        predicted_artist = encoder_Artist.inverse_transform([predict_gen[0][index]])[0][0]
        acc_artist = (actual_artist == predicted_artist)
        artist_weight  = artistsWeightTable.loc[actual_artist].Weight
        actual_type = AWTable.loc[imgId].Type_all
        actual_mat = AWTable.loc[imgId].Material_all
        title = AWTable.loc[imgId].Title_all
        description = AWTable.loc[imgId].Description_all       
        predTable.loc[index] = [imgId, predicted_artist, actual_artist, acc_artist, artist_weight, predicted_year, actual_year, err_year, actual_type, actual_mat, title, description]
   
    predTable.to_csv(LOGS_FOLDER + 'Predictions.csv', index=False, encoding='utf8')
    print('***** Predictions on test set saved in ' + LOGS_FOLDER + 'Predictions.csv')
    
# Custom data generator to provide batch of labelized data
def data_generator(data_set, Train=False):
       
     iBatch = 0
     steps_per_epoch = ceil(data_set.shape[0] / BATCH_SIZE)
     
     while True:
          batch_Id = data_set.iloc[iBatch * BATCH_SIZE : (iBatch+1) * BATCH_SIZE].index
          batch_image = []
          batch_target_artist = [] 
          batch_artist_weight = []
          batch_target_year = []
          batch_target_type = []
          batch_target_mat = []

          for Id in batch_Id:
              artist = AWTable.loc[Id].Artist
              year = AWTable.loc[Id].Year_Est
              types = AWTable.loc[Id].Type_all
              mats = AWTable.loc[Id].Material_all
              path = os.path.join(DIR_IMG, Id +".jpg")
              artist_weight  = artistsWeightTable.loc[artist].Weight

              #matplotlib.image.imsave(Id + '.jpg', np.array(image.load_img(path)))

              img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
              img = np.array(img)

              # apply model specific pre-processing
              if(MODEL_NAME == 'RESNET'):    img_prepro = applications.resnet50.preprocess_input(img)
              if(MODEL_NAME == 'Xception'):    img_prepro = applications.xception.preprocess_input(img)
              else: img_prepro = applications.vgg16.preprocess_input(img)
              batch_image.append(img_prepro)

              batch_target_artist.append(artist)
              batch_artist_weight.append(artist_weight)
              batch_target_year.append(year)
              batch_target_type.append(stringToList(types))
              batch_target_mat.append(stringToList(mats))
              if(Train):
                  # Basic Data Augmentation - Horizontal Flipping
                  flip_img = img
                  flip_img = np.array(flip_img)
                  flip_img = np.fliplr(flip_img)
                  # apply model specific pre-processing
                  if(MODEL_NAME == 'RESNET'):   flip_img_prepro = applications.resnet50.preprocess_input(flip_img)
                  if(MODEL_NAME == 'Xception'):    flip_img_prepro = applications.xception.preprocess_input(flip_img)
                  else: flip_img_prepro = applications.vgg16.preprocess_input(flip_img)
                  batch_image.append(np.array(flip_img_prepro))
          
                  # test - save img
                  #matplotlib.image.imsave('../../test/img/' + Id + '_' + artist +'_' + str(year) +'_resized.jpg', img_prepro)
                  # test - save img
                  #matplotlib.image.imsave('../../test/img/' + Id + '_' + artist +'_' +  str(year) +'_resized_flipped.jpg', flip_img_prepro)

                  batch_target_artist.append(artist)
                  batch_artist_weight.append(artist_weight)
                  batch_target_year.append(year)
                  batch_target_type.append(stringToList(types))
                  batch_target_mat.append(stringToList(mats))

          batch_image = np.array( batch_image) # / 255 # remove if using model pre-processing
          
          batch_target_artist = np.array( batch_target_artist ).reshape(-1,1)
          batch_target_artist = encoder_Artist.transform(batch_target_artist)
          #batch_artist_weight = np.array( batch_artist_weight).reshape(-1,1)
          # batch_artist is concatenation of artist weight (first column) and artist values one hot encoded (remaining columns)
          #batch_artist = np.append(batch_artist_weight, batch_target_artist, axis = 1)
          batch_artist_weight = np.array(batch_artist_weight)
          batch_artist = batch_target_artist
          
          batch_target_type = encoder_Type.transform(batch_target_type)
          batch_target_mat = encoder_Mat.transform(batch_target_mat)          
          batch_target_year = np.array( batch_target_year ).reshape(-1,1)
          batch_target_year = year_scaler.transform(batch_target_year)
          
          # if training only return tuple image + dictionary with correct oututs
          #if(Train):
          #yield(batch_image, {'artist': batch_artist, 'year': batch_target_year,'type': batch_target_type, 'mat': batch_target_mat} )
          # if test or validation also return a sample weight for the metrics evaluation
         # else:
          yield(batch_image, {'artist': batch_artist, 'year': batch_target_year,'type': batch_target_type, 'mat': batch_target_mat}, {'artist': batch_artist_weight} )
          iBatch += 1
          if(iBatch == steps_per_epoch): iBatch = 0 
          
          
def stringToList(text):
    types = []
    text_type = re.split(";", text)
    for typ in text_type:
        if not typ == '':
            types.append(typ)  
    return types

################# MAIN
    
def MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 512,  n_epochs = 40, batch_s = 20, img_s = 200, descr = 'No info'):

    global dic_artist_weights, LOGS_FOLDER, MODEL_FOLDER, DIR_IMG, AWTable, artistsWeightTable, encoder_Artist, encoder_Type, encoder_Mat, year_scaler, nClassesArtist, nClassesType, nClassesMat, train_set, valid_set, test_set, NUM_EPOCHS, BATCH_SIZE, IMG_SIZE, MODEL_NAME, NUM_HIDDEN_UNITS, RUN_DESCR

    #DIR_IMG = '../../data/1_original/img/'
    DIR_IMG = '../../data/TOP100/original/' #must change to previous
    DIR_METADATA = '../../data/2_meta_files/'
    DIR_LOGS_BASE = '../../logs/' 
    MODEL_FOLDER = '../../models/'
    NUM_EPOCHS = n_epochs
    BATCH_SIZE = batch_s
    IMG_SIZE = img_s
    MODEL_NAME = model
    NUM_HIDDEN_UNITS = h_units
    RUN_DESCR = descr
    
    datetime_object = datetime.datetime.now()
    LOGS_FOLDER = DIR_LOGS_BASE  + str(datetime_object.day) + '-' + str(datetime_object.month) + '_' + str(datetime_object.hour) + '.' + str(datetime_object.minute) + '_' + input_file + '_' + MODEL_NAME + str(h_units) + '_EPOCH' +  str(NUM_EPOCHS) + '_BS' + str(BATCH_SIZE) + '_IS'+ str(IMG_SIZE) + '/'
    
    # writing to log file
    os.mkdir(LOGS_FOLDER)
    f = open(LOGS_FOLDER + MODEL_NAME + '_info.txt', 'a')
    msg1 = 'run description = \t{} \ninput_file = \t{} \nmodel = \t{} \nh_units = \t{} \nn_epochs = \t{} \nbatch_s = \t{} \nimg_s = \t{} \n\n'.format(RUN_DESCR, input_file, MODEL_NAME, NUM_HIDDEN_UNITS, NUM_EPOCHS, BATCH_SIZE, IMG_SIZE)
    print(msg1)    
    f.write(msg1)

    #u.get_IMG_size_statistics(DIR)
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
    f.write(msg2)
    f.close()
    
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
    #artistsWeightTable = 1 / artistsFreqTable / nClassesArtist
    
    # Using a min max scaler (0-1) for Year
    train_valid_years = pd.concat([train_set.Year_Est, valid_set.Year_Est], axis = 0)
    year_scaler = MinMaxScaler()
    year_scaler.fit(np.asarray(train_valid_years).reshape(-1,1))
    
    # do exactly the same as artistsWeightTable !!!
    from sklearn.utils.class_weight import compute_class_weight
    a = AWTable['Artist'].to_numpy(copy=True).reshape(-1,1)
    all_artists_encoded = encoder_Artist.transform(a)
    artist_integers = np.argmax(all_artists_encoded, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(artist_integers), artist_integers)
    dic_artist_weights = dict(enumerate(class_weights))

    #debugMetrics()
    #debugDataGen()

    print("*****    2. Learning")
    learnModelMulti()
    u.makePlots(LOGS_FOLDER, MODEL_NAME)
    
    print("*****    3. Evaluating on test set")
    testModelMulti()
    
    print("*****    4. Predicting on test set")
    predictModelMulti()    
    
def debugDataGen():
        # ======== CODE to DEBUG the generator =======================================
    train_gen = data_generator(train_set, True)
    valid_gen = data_generator(valid_set, False)
    test_gen = data_generator(test_set, False)
    steps_per_epoch = ceil(valid_set.shape[0] / BATCH_SIZE)
    (batch_image_t, dicto_t) = next(train_gen)
    (batch_image, dicto, weights) = next(valid_gen)
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
    
    train_gen = data_generator(train_set)
    valid_gen = data_generator(valid_set)
    test_gen = data_generator(test_set)
    steps_per_epoch = ceil(valid_set.shape[0] / BATCH_SIZE)
    (batch_image, dicto) = next(train_gen)

    # testing weighted acc for Artist
    y_true = dicto.get('artist')
    y_pred = np.copy(y_true[:,1:])
    # change the first 3 lines so that they are incorrect
    for i in range(0, 3):
        trueidx = np.argmax(y_pred[i])
        if trueidx == 0:
            newidx = trueidx + 1
        else:
            newidx = trueidx - 1
        y_pred[i][trueidx] = 0
        y_pred[i][newidx] = 1
 
    acc_abs = cm.accuracy_abs(y_true,y_pred).eval(session=K.get_session())
    acc_weighted = cm.accuracy_w(y_true,y_pred).eval(session=K.get_session())    

    #cc_abs = cm.categorical_crossentropy_abs(y_true,y_pred).eval(session=K.get_session())    
    #cc_weighted = cm.accuracy_w(y_true,y_pred).eval(session=K.get_session())
    
    # testing mae with tol for Year
    weights = y_true[:,0]
    y_true = y_true [:,1:]
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
    
#MLT_learn_test(input_file = 'AWTableSetA.csv', model = 'RESNET', h_units = 512,  n_epochs = 1, batch_s = 10, img_s = 200)

# To DO 
# stemming to material
# dictionary: add last name of artist
# run on full dataset
# 2. Use Keras data augmentation
# try online data augmentation
# implement step decay

#inverted = encoder_Artist.inverse_transform([dicto.get('artist')[1]])[0][0]
#encoder_Artist.inverse_transform([[0,0,0,0,1]])[0][0]

#   Code to use Keras built in data augm
#    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
#                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
#                             horizontal_flip=True, fill_mode="nearest")
    

 
