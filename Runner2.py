#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:33:55 2019

@author: admin
"""


import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; 
## The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="3"; 

from MLT_Cust import MLT_learn_test, MLT_test

MLT_test(input_file = 'AWTableTOP100.csv', model_foler = '../../models/15-5_16.6_AWTableTOP100.csv_RESNET_Un5000_FOCALTrue_WEIGHTINGTrue_WS0.2/', model = 'RESNET', w_smooth = 0.2, batch_size = 100)

#MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 5000,  n_epochs = 80, batch_s = 100, img_s = 224, dropout = 0.5, focal = True, weighting = True, w_smooth = 0.2, descr = 'BN + adam  focal loss + smoothed weight, 1 dropout 0.5 + L2 reg artist + no year')