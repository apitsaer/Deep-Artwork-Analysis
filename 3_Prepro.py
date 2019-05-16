#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:53:49 2019

@author: Alexandre Pitsaer
"""
import os, shutil
import matplotlib as plt
import seaborn as sns
import pandas as pd
import Utils as u
import re

import warnings
warnings.filterwarnings('ignore')

SOURCE_DIR = "../../data/1_original/img"
SOURCE_TABLE = "../../data/2_meta_files/AWTable.csv"
#DEST_DIR = "../../data/TestSetA/original"
DEST_TABLE = "../../data/2_meta_files/AWTableTOP100.csv"

def generateArtistsold():
    for row in AWTable.itertuples():
        if(len(row.ArtistRaw.split(":")) >1):
            AWTable.set_value(row.Index, 'Artist', row.ArtistRaw.split(":")[1])

# calculate Artist based on  ArtistRaw
def generateArtists():

    uknList = ["anoniem", " anoniem", "onbekend", " onbekend", "niet van toepassing", " niet van toepassing", "", " "]
    
    for row in AWTable.itertuples():
    
        if(len(row.Artist_all.split(":")) >1):
            AWTable.at[row.Index, 'Artist'] = row.ArtistRaw.split(":")[1]          

        # fill Artist with "u" when unknown
        if(row.Artist in uknList):
            AWTable.at[row.Index, 'Artist'] = 'u'

    #ArtistCount = AWTable.groupby(['Artist']).size().reset_index(name='counts') #6 623 artist after cleanup unknown under 1 name = 'u
    #ArtistCount = ArtistCount[ArtistCount.counts >= 10]   
    #filterA = AWTable["Artist"].isin(ArtistCount.Artist) 
       
def createSubsetTop(x):
    ArtistCount = AWTable.groupby(['Artist']).size().reset_index(name='counts') #6 623 artist after cleanup unknown under 1 name = 'u
    ArtistCount = ArtistCount[ArtistCount.Artist != 'u']   
    Top100Artist = ArtistCount.nlargest(x, 'counts')
    filterA = AWTable["Artist"].isin(Top100Artist.Artist) 
    return AWTable[filterA]
    

def createSubsetA(selArt):    
    mask = AWTable["Artist"].isin(selArt) 
    AWTableFilt = AWTable[mask]
    return AWTableFilt
    
def moveFiles(AWTableFilt):
    source = os.listdir(SOURCE_DIR)
    #destination = DEST_DIR
    countfile = 0
    for fileName in source:
       #Id = fileName.split(".")[0]
       full_path = os.path.join(SOURCE_DIR, fileName)       
       if(not AWTableFilt[AWTableFilt['Id']+'.jpg' == fileName].empty):
           countfile += 1
           dest_path = os.path.join(DEST_DIR, fileName) 
           shutil.copy(full_path, dest_path)
    print("Copy files of, # files moved = " + str(countfile))
    
def generateYearEst():
    
    # creating Year data field with integer equal to mean value of interval
    AWTable.insert(3, 'Year_Est', 0)
    AWTable.insert(4, 'Year_Min', 0)
    AWTable.insert(5, 'Year_Max', 0)
    AWTable.Year_Est = AWTable.Year_Est.astype(float)
    for idx, row in AWTable.iterrows():
        year_est = 0
        #year_string = row.Year.replace(' ','')
        year_string = row.Year
        print(idx)
        
        if(idx == 11066):
            print('stop')
        
        # if it contains key work 'before'
        if(year_string == 'u' or year_string == ''):
            print('unkmown year, Id = {} , Artist = {}'.format(row.Id, row.Artist) )
            year_est = 1700
        else:            
            if('before' in year_string or 'after' in year_string or 'ca' in year_string or 'na' in year_string or 'voor' in year_string):
                #year_split = [s for s in year_string.split(' - ') if s.isdigit()]
                year_split = [s for s in re.split(' - |\. |\s|-', year_string) if s.isdigit()]
            else:
                year_split = year_string.split(' -  ')
            
            if(len(year_split) == 1 or (len(year_split) == 2 and year_split[1]=='') ): 
                if('-' in year_split[0] and year_split[0][0] != '-'):
                    year_clean = int(year_split[0].split('-')[0])
                else:
                    year_clean = int(year_split[0])
                year_min = year_clean
                year_max = year_clean
            elif(len(year_split) == 2): 
                # conatining a nother - but not at beginning (negative years)
                if('-' in year_split[0] and year_split[0][0] != '-'):
                    year_min = int(year_split[0].split('-')[0])
                else:
                    year_min = int(year_split[0])
                if('-' in year_split[1] and year_split[1][0] != '-'):
                    year_max = int(year_split[1].split('-')[0])
                else:
                    year_max = int(year_split[1])                    
            elif(len(year_split) == 4): 
                year_min = int(year_split[0])
                year_max = int(year_split[2])
#            elif(len(year_split) == 6): 
#                year_max = int(year_split[0])
#                year_max = int(year_split[3])
            
            if(year_min > 2020): year_min /= 10
            if(year_max > 2020): year_max /= 10
               # year_min = year_split[0]
                #year_max = year_split[1]
            year_est = (year_min + year_max) / 2
        #else: year_clean = 1700 # TO DO : also throw an exception
        AWTable.set_value(idx, 'Year_Est', year_est)
        AWTable.set_value(idx, 'Year_Min', year_min)
        AWTable.set_value(idx, 'Year_Max', year_max)
    
##################### MAIN #####################

AWTable = pd.read_csv(SOURCE_TABLE, keep_default_na=True)
AWTable = AWTable.fillna('u')
AWTable.columns = AWTable.columns.str.strip() #remove leading and trailing white space if any

#print("# Artists RAW = " + str(AWTable['Artist'].nunique()))
generateArtists()
print("# Artists Cleaned (incl. 1 unknown) = " + str(AWTable['Artist'].nunique()))

generateYearEst()
      
    
# Showing some STATS
(all_Types, all_Mats) = u.getAllTypeMat(AWTable)
nClassesArtist = AWTable['Artist'].nunique()
nClassesType = len(all_Types)
nClassesMat = len(all_Mats)

print('Found {} unique Artist'.format(nClassesArtist))
print('Found {} unique Type'.format(nClassesType))
print('Found {} unique Material'.format(nClassesMat))
          
# Generate subset based on list of Artists
selArt = ['Aldegrever, Heinrich','Beham, Hans Sebald','Bemme, Joannes', 'Woodbury & Page', 'Meissener Porzellan Manufaktur']
selArt = ['Marcus, Jacob Ernst', 'Cort, Cornelius']
AWTableFilt = createSubsetA(selArt)

# ALTERNATIVELY, generate subset based on top 
AWTableFilt = createSubsetTop(20)

AWTableFilt = AWTableFilt[['Id', 'Artist', 'Year_Est', 'Type_all', 'Material_all', 'Title_all', 'Description_all']]
#AWTableFilt.to_csv(DEST_TABLE, index=False, encoding='utf8')
#moveFiles(AWTableFilt)

print('{} unique Artist in subset'.format(AWTableFilt['Artist'].nunique()))
print('{} Rows in subset out of {}, proportion = {:{prec}}'.format(AWTableFilt.shape[0], AWTable.shape[0], AWTableFilt.shape[0] / AWTable.shape[0],prec='.3'))



