#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:24:08 2019

@author: admin
"""

import pandas as pd


DIR = '../../data/TOP20/'
INPUT_FILE = 'AWTableTOP20.csv'
SOURCE_TABLE = DIR + INPUT_FILE

AWTable = pd.read_csv(SOURCE_TABLE, keep_default_na=True)

# Showing some STATS
nClassesArtist = AWTable['Artist'].nunique()
artistsTable = AWTable['Artist'].value_counts(normalize=False)

print('# AW for Luyken, Jan = '  + str(artistsTable['Luyken, Jan']))
print('# AW for Galle, Philips = '  + str(artistsTable['Galle, Philips']))

dfArtists = pd.DataFrame([artistsTable]).transpose()

print('Found {} unique Artist'.format(nClassesArtist))

