#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:33:55 2019

@author: admin
"""

import shutil, os
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="3"; 

from MLT_Cust import MLT_learn_test

#DIR_LOGS_BASE = '../../logs/'
#
#
#for currentFolder in os.listdir(DIR_LOGS_BASE):
#    if not str(currentFolder).startswith('.'):
#        shutil.rmtree(DIR_LOGS_BASE + currentFolder)


MLT_learn_test(input_file = 'AWTableSetAL.csv', model = 'RESNET', h_units = 768,  n_epochs = 1, batch_s = 50, img_s = 224, descr = 'test adam custom matrix weight')
MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 3000,  n_epochs = 50, batch_s = 50, img_s = 200, descr = 'adam custom matrix weight, 1 dropout + no year')
#MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 1000,  n_epochs = 40, batch_s = 50, img_s = 224, descr = 'rmsprop std keras with only sample weights artist')
#MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 6000,  n_epochs = 50, batch_s = 50, img_s = 224, descr = 'adam std keras with only sample weights artist')

#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 1200,  n_epochs = 50, batch_s = 50, img_s = 200, descr = 'with MLT Cust. and categorical_crossentropy_abs')

#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'Xception', h_units = 512,  n_epochs = 40, batch_s = 50, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'Xception', h_units = 768,  n_epochs = 40, batch_s = 50, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 4096,  n_epochs = 40, batch_s = 50, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'Xception', h_units = 1024,  n_epochs = 50, batch_s = 20, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'VGG', h_units = 1024,  n_epochs = 50, batch_s = 20, img_s = 200)

#MLT_learn_test(input_file = 'AWTableSetA.csv', model = 'RESNET', h_units = 512,  n_epochs = 20, batch_s = 50, img_s = 200)
#MLT_learn_test(input_file = 'AWTableSetA.csv', model = 'RESNET', h_units = 512,  n_epochs = 2, batch_s = 20, img_s = 200)

#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 512,  n_epochs = 60, batch_s = 20, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 768,  n_epochs = 50, batch_s = 20, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 1024,  n_epochs = 50, batch_s = 20, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 768,  n_epochs = 50, batch_s = 50, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 768,  n_epochs = 50, batch_s = 20, img_s = 350)



