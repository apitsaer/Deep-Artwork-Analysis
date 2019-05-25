#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:33:55 2019

@author: admin
"""


import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; 
## The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; 

from MLT_Cust import MLT_learn_test, MLT_test



#MLT_test(input_file = 'AWTableTOP100.csv', model_foler = '../../models/24-5_11.0_AWTableTOP100.csv_RESNET_Un6000_FOCALFalse_WEIGHTINGTrue_WS0.02/', model = 'RESNET', w_smooth = 0.02, batch_size = 100)


# the 4 tests RUN
#MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 6000,  n_epochs = 100, batch_s = 120, img_s = 224, dropout = 0.5, focal = True, weighting = True, w_smooth = 0.1, descr = 'BN + adam  focal loss + smoothed weight, 1 dropout 0.5 + L2 reg artist + no year')
#MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 6000,  n_epochs = 100, batch_s = 100, img_s = 224, dropout = 0.5, focal = False, weighting = False, w_smooth = 0.1, descr = 'BN + adam  focal loss + smoothed weight, 1 dropout 0.5 + L2 reg artist + no year')
MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 6000,  n_epochs = 100, batch_s = 100, img_s = 224, dropout = 0.5, focal = False, weighting = True, w_smooth = 0.02, descr = 'BN + adam  NO focal loss + smoothed weight, 1 dropout 0.5 + L2 reg artist + no year')
MLT_learn_test(input_file = 'AWTableTOP100.csv', model = 'RESNET', h_units = 6000,  n_epochs = 100, batch_s = 100, img_s = 224, dropout = 0.5, focal = False, weighting = True, w_smooth = 0.1, descr = 'BN + adam  NO focal loss + NO weighting, 1 dropout 0.5 + L2 reg artist + no year')

# try less smoothing
# try with year again
# implement MAP
# try Resnetx

##MLT_learn_test(input_file = 'AWTableTOP20.csv', model = 'RESNET', h_units = 1500,  n_epochs = 60, batch_s = 50, img_s = 224, dropout = 0.5, focal = True, weighting = True, w_smooth = 0.2, descr = 'no BN + adam  focal loss + smoothed weight, 1 dropout 0.5 + L2 reg artist + no year')
