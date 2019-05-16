#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:11:22 2019

@author: admin
"""
import re
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def getAllTypeMat(AWTableTOP):

    all_type = set()
    all_mat = set()
    
    for row in AWTableTOP.itertuples():
        text_mat = re.split(";", row.Material_all)
        for mat in text_mat:
            if not mat == '':
                all_mat.add(mat)
    
        text_type = re.split(";", row.Type_all)
        for typ in text_type:
            if not typ == '':
                all_type.add(typ)
        
    return (all_type, all_mat)

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

    INPUT_FILE = DIR + MODEL + '.log'
    OUTPUT_FILE = DIR +  MODEL + '_train_valid_metrics.png'
    OUTPUT_FILE_L1 = DIR +  MODEL + '_train_valid_loss.png'
    OUTPUT_FILE_LS = DIR +  MODEL + '_train_valid_losses_split.png'
    
    train_logs = pd.read_csv(INPUT_FILE, keep_default_na=False)
    
    plt.figure(0)
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axs[0, 0].plot(train_logs.epoch, smooth_curve(train_logs.artist_accuracy_w), 'b', label='Train')
    axs[0, 0].plot(train_logs.epoch, smooth_curve(train_logs.val_artist_accuracy_w), 'r', label='Val')
    axs[0, 0].set_title('Artist')
    axs[0, 0].set_ylabel('Avg Class Acc')
    axs[0, 0].legend()
    axs[0, 0].grid()
    
    axs[0, 1].plot(train_logs.epoch, smooth_curve(train_logs.year_mean_absolute_error), 'b', label='Train')
    axs[0, 1].plot(train_logs.epoch, smooth_curve(train_logs.val_year_mean_absolute_error), 'r', label='Val')
    axs[0, 1].set_title('Year')
    axs[0, 1].set_ylabel('MAE')
    axs[0, 1].legend()
    axs[0, 1].grid()
    
    axs[1, 0].plot(train_logs.epoch, smooth_curve(train_logs.type_precision), 'b', label='Train')
    axs[1, 0].plot(train_logs.epoch, smooth_curve(train_logs.val_type_precision), 'r', label='Val')
    axs[1, 0].set_title('Type')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].legend()
    axs[1, 0].grid()
    
    axs[1, 1].plot(train_logs.epoch, smooth_curve(train_logs.mat_precision), 'b', label='Train')
    axs[1, 1].plot(train_logs.epoch, smooth_curve(train_logs.val_mat_precision), 'r', label='Val')
    axs[1, 1].set_title('Material')
    axs[1, 1].set_ylabel('Precision')
    axs[1, 1].legend()
    axs[1, 1].grid()
    
    #fig.tight_layout() 
      
    for ax in axs.flat:
        ax.set(xlabel='Epoch')
    ## Hide x labels and tick labels for top plots
    #for ax in axs.flat:
    #    ax.label_outer()
    
    fig.suptitle('Train & Val Metrics')
    
    #plt.subplots_adjust(top=1.2, bottom=0.2, left=0.10, right=0.95, hspace=0.45, wspace=0.35)
    plt.savefig(OUTPUT_FILE)
    plt.show()

    plt.figure(1)
    fig, axs = plt.subplots(2, 2,  constrained_layout=True)
    axs[0, 0].plot(train_logs.epoch, smooth_curve(train_logs.artist_loss), 'b', label='Train')
    axs[0, 0].plot(train_logs.epoch, smooth_curve(train_logs.val_artist_loss), 'r', label='Val')
    axs[0, 0].set_title('Artist')
    axs[0, 0].set_ylabel('Cat x entropy weighted')
    axs[0, 0].legend()
    axs[0, 0].grid()
    
    axs[0, 1].plot(train_logs.epoch, smooth_curve(train_logs.year_loss), 'b', label='Train')
    axs[0, 1].plot(train_logs.epoch, smooth_curve(train_logs.val_year_loss), 'r', label='Val')
    axs[0, 1].set_title('Year')
    axs[0, 1].set_ylabel('MAE')
    axs[0, 1].legend()
    axs[0, 1].grid()
    
    axs[1, 0].plot(train_logs.epoch, smooth_curve(train_logs.type_loss), 'b', label='Train')
    axs[1, 0].plot(train_logs.epoch, smooth_curve(train_logs.val_type_loss), 'r', label='Val')
    axs[1, 0].set_title('Type')
    axs[1, 0].set_ylabel('Bin x entropy')
    axs[1, 0].legend()
    axs[1, 0].grid()
    
    axs[1, 1].plot(train_logs.epoch, smooth_curve(train_logs.mat_loss), 'b', label='Train')
    axs[1, 1].plot(train_logs.epoch, smooth_curve(train_logs.val_mat_loss), 'r', label='Val')
    axs[1, 1].set_title('Material')
    axs[1, 1].set_ylabel('Bin x entropy')
    axs[1, 1].legend()
    axs[1, 1].grid()
   
    #fig.tight_layout() 
      
    for ax in axs.flat:
        ax.set(xlabel='Epoch')
    ## Hide x labels and tick labels for top plots
    #for ax in axs.flat:
     #   ax.label_outer()
    
    #plt.subplots_adjust(top=1.2, bottom=0.2, left=0.10, right=0.95, hspace=0.65, wspace=0.55)
    fig.suptitle('Train & Val Losses')
    plt.savefig(OUTPUT_FILE_LS)
    plt.show()
    
    plt.figure(2)
    plt.plot(train_logs.epoch, smooth_curve(train_logs.loss), 'b', label='Train')
    plt.plot(train_logs.epoch, smooth_curve(train_logs.val_loss), 'r', label='Val')
    plt.title('Total Loss')
    plt.ylabel('Multi-task sum weighted loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT_FILE_L1)
    plt.show()

# get some statistics on the original image size in order to determine acceptable resizing
def get_IMG_size_statistics(DIR):
    heights = []
    widths = []
    directory = DIR + 'original'
    for img in os.listdir(directory): 
        if(img.endswith('jpg')):
            path = os.path.join(directory, img)
            data = image.load_img(path)
            data = image.img_to_array(data)
            heights.append(data.shape[0])
            widths.append(data.shape[1])
    print(img)
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))
    
def save_CM_Table(DIR, MODEL, y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    cmNorm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cm_df = pd.DataFrame(cm)
    cmNorm_df = pd.DataFrame(cmNorm)
    classes_df = pd.DataFrame(unique_labels(y_true, y_pred))
    #classes_dfT = pd.concat(['', classes_df.transpose()], axis = 1)
    
    cm_df = pd.concat([classes_df, cm_df], axis = 1)
    cmNorm_df = pd.concat([classes_df, cmNorm_df], axis = 1)
    
    cm_df.to_csv(DIR +  MODEL + '_confusion_matrix' + '.csv', index=True, encoding='utf8')
    cmNorm_df.to_csv(DIR +  MODEL + '_confusion_matrix_norm' + '.csv', index=True, encoding='utf8')


def plot_confusion_matrix(y_true, y_pred, 
                          #classes,
                          DIR, MODEL,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Artists confusion matrix, normalized'
        else:
            title = 'Artists confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def save_confusion_matrix(DIR, MODEL, y_true, y_pred):

    np.set_printoptions(precision=2)
    OUTPUT_FILE_1 = DIR +  MODEL + '_confusion_matrix_abs'
    OUTPUT_FILE_2 = DIR +  MODEL + '_confusion_matrix_norm'
    
    plt.figure(1)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_true, y_pred, DIR, MODEL,
                          #classes=class_names,
                          title='Artist confusion matrix, without normalization')
    
    plt.savefig(OUTPUT_FILE_1 + '.png')
    plt.show()

    plt.figure(2)
    # Plot normalized confusion matrix
    plot_confusion_matrix(y_true, y_pred, DIR, MODEL,
                          #classes=class_names, 
                          normalize=True,
                          title='Artist confusion matrix, normalized')
    
    plt.savefig(OUTPUT_FILE_2 + '.png')
    plt.show()

#  import some data to play with
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#class_names = iris.target_names
## Split the data into a training set and a test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
## Run classifier, using a model that is too regularized (C too low) to see
## the impact on the results
#classifier = svm.SVC(kernel='linear', C=0.01)
#y_pred = classifier.fit(X_train, y_train).predict(X_test)
#
#save_confusion_matrix('/Users/admin/Documents/AWImpl/logs/', 'RESNET', y_test, y_pred)
#save_CM_Table('/Users/admin/Documents/AWImpl/logs/', 'RESNET', y_test, y_pred)

#makePlots('/Users/admin/Documents/AWImpl/logs/', 'RESNET')