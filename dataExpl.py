#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:10:49 2019

@author: admin
"""

import pandas as pd
import re

def geneRateSubsetArtists():
    
    selArt = [' Aldegrever, Heinrich',' Beham, Hans Sebald',' Bemme, Joannes', ' Woodbury & Page']
    mask = AWTable["Artist"].isin(selArt) 
    AWTableFilt = AWTable[mask]
    AWTableFilt['Artist'].value_counts(normalize=False)
    AWTableFilt['IdShort'].nunique()
    AWTableFilt.to_csv("../data/TestRDArtists/AWTable4.csv", index=False, encoding='utf8')


AWTable = pd.read_csv('../data/AWTable.csv', keep_default_na=False)
AWTable['Type'].value_counts(normalize=True)
#all_types = AWTable['Type'].value_counts(normalize=True)
#all_mat = AWTable['Material'].value_counts(normalize=True)
AWTable['Type'].nunique()
AWTable['Material'].nunique()
AWTable['Material'].value_counts(normalize=True)

all_types = set()
all_mat = set()

for row in AWTable.itertuples():
    text_mat = re.split(";", row.Material_all)
    for mat in text_mat:
        all_types.add(mat)

    text_type = re.split(";", row.Type_all)
    for typ in text_type:
        all_mat.add(typ)

mat_dict = {}
countRowSimWord_mat = 0
for row in AWTable.itertuples():
    text_mat = re.split("[\s;,.-]", row.Material_all)
    text_type = re.split("[\s;,.-]", row.Type_all)
    text_descr = re.split("[\s;,.-]", row.Description_all)
    text_tit = re.split("[\s;,.-]", row.Title_all)
    for word_mat in text_mat:
        found_mat_descr = False
        found_mat_tit = False
        for word_descr in text_descr:
            # also check that word is not a stop word
            if (word_mat == word_descr and not (word_mat == "" or word_mat=="en" or word_mat=="in" or word_mat=="de"  or word_mat=="het" or word_mat=="een")):
                found_mat_descr = True
        for word_tit in text_tit:
            # also check that word is not a stop word
            if (word_mat == word_tit and not (word_mat == "" or word_mat=="en" or word_mat=="in" or word_mat=="de"  or word_mat=="het" or word_mat=="een")):
                found_mat_tit = True
                
        if found_mat_descr or found_mat_tit:
            countRowSimWord_mat += 1
            mat_dict[row.IdShort] = word_mat
            #if found_mat_descr:
                #print("same word MAT found in DESCR: " + word_mat + " in Id: " + str(row.IdShort))
            #if found_mat_tit:
                #print("same word MAT found in TITLE: " + word_mat + " in Id: " + str(row.IdShort))

tot = AWTable.shape[0] * 1.0
ratio_mat = len(mat_dict)/tot


type_dict = {}
countRowSimWord_type = 0
for row in AWTable.itertuples():
    text_type = re.split("[\s;,.-]", row.Type_all)
    text_descr = re.split("[\s;,.-]", row.Description_all)
    text_tit = re.split("[\s;,.-]", row.Title_all)
    for word_type in text_type:
        found_type_descr = False
        found_type_tit = False
        for word_descr in text_descr:
            #print("word_mat: "+ word_mat + "and word_descr: " + word_descr)
            if (word_type == word_descr and not (word_type == "" or word_type=="en" or word_type=="in" or word_type=="de"  or word_type=="het" or word_type=="een")):
                found_type_descr = True
        for word_tit in text_tit:
            # also check that word is not a stop word
            if (word_type == word_tit and not (word_type == "" or word_type=="en" or word_type=="in" or word_type=="de"  or word_type=="het" or word_type=="een")):
                found_type_tit = True
                
        if found_type_descr or found_type_tit:
            countRowSimWord_type += 1
            type_dict[row.IdShort] = word_type
            #if found_type_descr:
                #print("same word TYPE found in DESCR: " + word_type + " in Id: " + str(row.IdShort))
            #if found_type_tit:
                #print("same word TYPE found in TITLE: " + word_type + " in Id: " + str(row.IdShort))

ratio_type = len(type_dict)/tot

typesFoundinTxt = set(type_dict.values())
matsFoundinTxt = set(mat_dict.values())

typesFound_Ids = set(type_dict.keys())
matsFound_Ids = set(mat_dict.keys())

interectFound_Ids = typesFound_Ids.intersection(matsFound_Ids)
unionFound_Ids = typesFound_Ids.union(matsFound_Ids)

mat_count = {}
mat_list = mat_dict.values()
for mat in mat_list:
    if mat not in mat_count:
        mat_count[mat] = 1
    else:
        mat_count[mat] += 1
        
type_count = {}
type_list = type_dict.values()
for typ in type_list:
    if typ not in type_count:
        type_count[typ] = 1
    else:
        type_count[typ] += 1
        
print('\n *********')
print('{:.2%}'.format(len(unionFound_Ids)/tot) + ' of AWs have  common word in "type" or "material" and free text field ("title" or "description")')
print('\n ****  Mat:')
print('Material: Found ' + str(len(mat_dict)) + ' AWs with common word in material and free text field, corresp. to ' + '{:.2%}'.format(ratio_mat) + ' of total rows')
print('Max occ: ' +  str(max(mat_count.values())) )
print('Min occ: ' +  str(min(mat_count.values())) )
print('Avg occ: ' +  str(round(sum(mat_count.values())/len(mat_count),1)) )
print('Found # : ' +  str(len(mat_count)) + ' out of : ' + str(len(all_mat)) )

print('\n ****  Type:')
print('Type: Found ' + str(len(type_dict)) + ' AWs with common word in type and free text field, corresp. to ' + '{:.2%}'.format(ratio_type) + ' of total rows')
print('Max occ: ' +  str(max(type_count.values())) )
print('Min occ: ' +  str(min(type_count.values())) )
print('Avg occ: ' +  str(round(sum(type_count.values())/len(type_count),1)))
print('Found # : ' +  str(len(type_count)) + ' out of : ' + str(len(all_types)) )
