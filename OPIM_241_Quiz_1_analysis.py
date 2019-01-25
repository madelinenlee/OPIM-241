#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:08:41 2019

@author: madeline
"""


import pandas as pd
import numpy as np

boston_housing_data = pd.read_csv('/Users/madeline/Desktop/SPRING 2019/OPIM 241/Quiz 1/boston_housing.csv')
attributes = boston_housing_data.columns.tolist()

for attribute in attributes:
    print(boston_housing_data[attribute].describe())

    
def missing_values(data_frame):
    attributes = data_frame.columns.tolist()
    final_missing = pd.DataFrame(columns = attributes)
    for item in attributes:
        missing_data = data_frame[data_frame[item].isnull() == True]
        final_missing = final_missing.append(missing_data)
        final_missing = final_missing.drop_duplicates()
    print("number of rows with missing values: " + str(final_missing.shape[0]))
   
    return (final_missing)

boston_housing_missing = missing_values(boston_housing_data)

for item in attributes:
    boston_housing_data[item].hist()
    plt.title(item)
    plt.savefig(item+'_hist.png')
    plt.show()
