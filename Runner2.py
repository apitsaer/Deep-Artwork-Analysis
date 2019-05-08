#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:33:55 2019

@author: admin
"""

from MLT_Cust import MLT_learn_test


MLT_learn_test(input_file = 'AWTableSetAL.csv', model = 'RESNET', h_units = 768,  n_epochs = 1, batch_s = 50, img_s = 224, descr = 'test adam custom matrix weight')
MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 3000,  n_epochs = 50, batch_s = 50, img_s = 224, descr = 'adam custom matrix weight, 1 task + multiple dropout')


#MLT_learn_test(input_file = 'AWTableSetA.csv', model = 'RESNET', h_units = 768,  n_epochs = 1, batch_s = 50, img_s = 224, descr = 'adam custom matrix weight, year scaled + tol of 0.1')
#MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 2000,  n_epochs = 25, batch_s = 50, img_s = 224, descr = 'adam custom matrix weight, year scaled')

#MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 4096,  n_epochs = 60, batch_s = 50, img_s = 224, descr = 'with MLT Cust. and categorical_crossentropy_abs')
#MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 2048,  n_epochs = 60, batch_s = 50, img_s = 224, descr = 'with MLT Cust. and categorical_crossentropy_abs')
##MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'VGG', h_units = 64,  n_epochs = 40, batch_s = 50, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'VGG', h_units = 96,  n_epochs = 40, batch_s = 50, img_s = 200)
#MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'VGG', h_units = 128,  n_epochs = 40, batch_s = 50, img_s = 200)




