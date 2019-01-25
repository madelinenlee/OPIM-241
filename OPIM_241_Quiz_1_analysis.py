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

    