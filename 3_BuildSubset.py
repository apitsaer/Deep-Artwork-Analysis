#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:53:49 2019

@author: admin
"""
import os, shutil
import matplotlib as plt
import seaborn as sns
import pandas as pd

SOURCE_DIR = "../data/jpg2"
SOURCE_TABLE = "../data/AWTable.csv""
DEST_DIR = "../data/TestSetAT/original"

def generateArtistsold():
    for row in AWTable.itertuples():
        if(len(row.ArtistRaw.split(":")) >1):
            AWTable.set_value(row.Index, 'Artist', row.ArtistRaw.split(":")[1])

# calculate Artist based on  ArtistRaw
def generateArtists():
    for row in AWTable.itertuples():
        if(len(row.Artist_all.split(":")) >1):
            AWTable.at[row.Index, 'Artist'] = row.ArtistRaw.split(":")[1]          
        if(len(row.Artist2.split(":")) >1):
            AWTable.at[row.Index, 'Artist2'] = row.Artist2.split(":")[1]
        if(len(row.Artist3.split(":")) >1):
            AWTable.at[row.Index, 'Artist3'] = row.Artist3.split(":")[1]
        if(len(row.Artist4.split(":")) >1):
            AWTable.at[row.Index, 'Artist4'] = row.Artist4.split(":")[1]

# fill Artist with "u" when unknown        
def cleanArtists():
       
    uknList = ["anoniem", " anoniem", "onbekend", " onbekend", "niet van toepassing", " niet van toepassing", "", " "]
    for row in AWTable.itertuples():
        if(row.Artist in uknList):
            #Artisttmp = np.nan
            Artisttmp = 'u'
#            if(not row.Artist2 in uknList):
#                Artisttmp = row.Artist2
#            else:
#                if(not row.Artist3 in uknList):
#                    Artisttmp = row.Artist3
#                    print('level3')
#                else:
#                    if(not row.Artist4 in uknList):
#                        Artisttmp = row.Artist4    
#                        print('level4')
            AWTable.at[row.Index, 'Artist'] = Artisttmp

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
    destination = DEST_DIR
    countfile = 0
    for fileName in source:
       AWShortId = int(fileName.split("_")[0])
       full_path = os.path.join(SOURCE_DIR, fileName)       
       if(not AWTableFilt[AWTableFilt['IdShort']==AWShortId].empty):
           countfile += 1
           shutil.copy(full_path, destination)
    print("Copy files of, # files moved = " + str(countfile))
    
    
##################### MAIN

AWTable = pd.read_csv(SOURCE_TABLE, keep_default_na=False)

print("# Artists RAW = " + str(AWTable['Artist'].nunique()))
generateArtists()
cleanArtists()
print("# Artists Cleaned (incl. 1 unknown) = " + str(AWTable['Artist'].nunique()))
#selArt = ['Aldegrever, Heinrich','Beham, Hans Sebald','Bemme, Joannes', 'Woodbury & Page', 'Meissener Porzellan Manufaktur']
selArt = ['Aldegrever, Heinrich','Beham, Hans Sebald','Bemme, Joannes', 'Woodbury & Page']
AWTableFilt = createSubsetA(selArt)
AWTableFilt.to_csv("../data/TestSetAT/AWTableSetAT.csv", index=False, encoding='utf8')
moveFiles(AWTableFilt)
