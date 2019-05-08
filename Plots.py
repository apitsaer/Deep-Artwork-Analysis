#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:39:06 2019

@author: admin
"""
import pandas as pd
import matplotlib.pyplot as plt


def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

def makePlots(DIR, MODEL):

    #DIR = '../logs/'
    #MODEL = 'VGG_flatten'
    
    INPUT_FILE = MODEL + '.log'
    OUTPUT_FILE = MODEL + '.png'
    
    train_logs = pd.read_csv(DIR + INPUT_FILE, keep_default_na=False)
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(train_logs.epoch, smooth_curve(train_logs.artist_acc), 'b', label='Train')
    axs[0, 0].plot(train_logs.epoch, smooth_curve(train_logs.val_artist_acc), 'r', label='Val')
    axs[0, 0].set_title('Artist')
    axs[0, 0].set_ylabel('Acc')
    axs[0, 0].legend()
    
    axs[0, 1].plot(train_logs.epoch, smooth_curve(train_logs.year_mean_absolute_error), 'b', label='Train')
    axs[0, 1].plot(train_logs.epoch, smooth_curve(train_logs.val_year_mean_absolute_error), 'r', label='Val')
    axs[0, 1].set_title('Year')
    axs[0, 1].set_ylabel('MAE')
    axs[0, 1].legend()
    
    axs[1, 0].plot(train_logs.epoch, smooth_curve(train_logs.type_precision), 'b', label='Train')
    axs[1, 0].plot(train_logs.epoch, smooth_curve(train_logs.val_type_precision), 'r', label='Val')
    axs[1, 0].set_title('Type')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].legend()
    
    axs[1, 1].plot(train_logs.epoch, smooth_curve(train_logs.mat_precision), 'b', label='Train')
    axs[1, 1].plot(train_logs.epoch, smooth_curve(train_logs.val_mat_precision), 'r', label='Val')
    axs[1, 1].set_title('Material')
    axs[1, 1].set_ylabel('Precision')
    axs[1, 1].legend()
      
    for ax in axs.flat:
        ax.set(xlabel='Epoch')
    ## Hide x labels and tick labels for top plots
    for ax in axs.flat:
        ax.label_outer()
    
    plt.subplots_adjust(top=1.2, bottom=0.2, left=0.10, right=0.95, hspace=0.45, wspace=0.35)
    
    plt.savefig(DIR + OUTPUT_FILE, dpi=150)
    plt.show()

makePlots('RESNET')