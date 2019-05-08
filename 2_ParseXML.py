#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:49:35 2019

@author: admin
"""

# Step 2 of pre-processing: load all xml files in data frame. Keep "Artist" raw and process it later


import xml.etree.ElementTree as ET
import pandas as pd
import os

AWTable = pd.DataFrame(columns=["Id", "IdShort", "Artist","Year","Type","Material","Title","Description","Artist_all","Artist_raw","Type_all","Material_all","Title_all","Description_all","CountArtist","CountYear","CountType","CountMaterial","CountTitle","CountDescription","Year2","Year3"])
count = 0

MaxArtist = 1
MaxYear = 1
MaxType = 1
MaxMaterial = 1
MaxTitle = 1
MaxDescription = 1
 
def parseXmlFile(fullname):
    
    tree = ET.parse(fullname)
    r = tree.getroot()
    
    base = os.path.basename(fullname)
    Id = os.path.splitext(base)[0]
    IdShort = Id.split("_")[0]
  
    Artist_all = ''
    Type_all = ''
    Material_all = ''
    Title_all = ''
    Description_all = ''
    
    Artist_raw = ''

    Artist = ''
    Year = ''
    Year2 = ''
    Year3 = ''
    Type = ''
    Material = ''
    Title = ''
    Description = ''
    
    CountArtist = 0
    CountYear = 0
    CountType = 0
    CountMaterial = 0
    CountTitle = 0
    CountDescription = 0
    
    for creator in r.iter('creator'):
        if(isinstance(creator.text, str)):
            CountArtist += 1
            Artisttmp = creator.text.split(":")[1]
            if(Artisttmp[0]==" "): 
                Artisttmp = Artisttmp[1:]
            Artist_raw = Artist_raw + creator.text # keep raw text
            Artist_all = Artist_all + Artisttmp + ";" # cleaned version of text
            if(CountArtist == 1): Artist = Artisttmp
            global MaxArtist
            if(CountArtist > MaxArtist) : MaxArtist = CountArtist
    
    for title in r.iter('title'):
        if(isinstance(title.text, str)):
            CountTitle += 1
            Title_all = Title_all + title.text + ";"
            if(CountTitle == 1): Title = title.text
            global MaxTitle
            if(CountTitle > MaxTitle) : MaxTitle = CountTitle
 
    for description in r.iter('description'):
        if(isinstance(description.text, str)):
            Description = description.text
            CountDescription += 1
            Description_all = Description_all + description.text + ";"
            global MaxDescription
            if(CountDescription > MaxDescription) : MaxDescription = CountDescription
 
    for type in r.iter('type'):
        if(isinstance(type.text, str)):
            CountType += 1
            Type_all = Type_all + type.text + ";"
            if(CountType == 1): Type = type.text
            global MaxType
            if(CountType > MaxType) : MaxType = CountType
 
    for date in r.iter('date'):
        if(isinstance(date.text, str)):
            CountYear += 1
            if(CountYear == 1): Year = date.text
            if(CountYear == 2): Year2 = Artisttmp
            if(CountYear == 3): Year3 = Artisttmp
            global MaxYear
            if(CountYear > MaxYear) : MaxYear = CountYear
 
    for format in r.iter('format'):
        if(isinstance(format.text, str)):
            if(format.text.startswith('materiaal:')):
                CountMaterial += 1
                Materialtmp = format.text.split(":")[1]
                if(Materialtmp[0]==" "): 
                    Materialtmp = Materialtmp[1:]
                Material_all = Material_all + Materialtmp + ";"
                if(CountMaterial == 1): Material = Materialtmp
                global MaxMaterial
                if(CountMaterial > MaxMaterial) : MaxMaterial = CountMaterial
                     
    AWTable.loc[count] = [Id, IdShort, Artist, Year,Type,Material,Title,Description,Artist_all,Artist_raw,Type_all,Material_all,Title_all,Description_all,CountArtist,CountYear,CountType,CountMaterial,CountTitle,CountDescription,Year2,Year3]

def main():
    
    directory = '../data/1_original/xml'
    print("Parsing xml files in directory " + directory)
    for currentFile in os.listdir(directory):  
        if str(currentFile).endswith('.xml'):
            global count
            if(count % 100 ==0):
                print("\t @ file: " + str(count))
            count = count + 1
            parseXmlFile(os.path.join(directory, currentFile))

    AWTable.to_csv("../data/2.meta_files/AWTable.csv", index=False, encoding='utf8')
    print("Max # Artist element / AW in xml = " +  str(MaxArtist))
    print("Max # Year element / AW in xml = " +  str(MaxYear))
    print("Max # Material element / AW in xml = " +  str(MaxMaterial))
    print("Max # Type element / AW in xml = " +  str(MaxType))
    print("Max # Title element / AW in xml = " +  str(MaxTitle))
    print("Max # Description element / AW in xml = " +  str(MaxDescription))
    
main()
