#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:52:34 2019

@author: madeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pathname_in = '/Users/madeline/Downloads/flight-delays/flights.csv'
phx_pathname = '/Users/madeline/Downloads/flight-delays/phx.csv'

airline_dictionary = {'AA': '',
                      'NK': '',
                      'US': '',
                      'DL': '',
                      'WN': '',
                      'AS': '',
                      'F9': '',
                      'OO': '',
                      'HA': '',
                      'EV': '',
                      'B6': '',
                      'UA': '',
                      
        }


columns_to_drop = ['FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT',
                   'DESTINATION_AIRPORT','TAXI_OUT',
                   'WHEELS_OFF', 'SCHEDULED_TIME',
                   'ELAPSED_TIME', 'AIR_TIME',
                   'DISTANCE','WHEELS_ON', 'TAXI_IN',
                   'SCHEDULED_ARRIVAL', 'YEAR', 'MONTH']

departure_bins = np.array([0, 800, 1200, 1500, 1700, 2100])

def describe_attributes(data_frame):
    attributes = data_frame.columns.tolist()
    print(attributes)
    attributes.remove('AIRLINE')
    
    for a in attributes:
        print(data_frame[a].describe())
        data_frame[a].hist()
        plt.title(a)
        plt.show()
        data_frame.plot.scatter(a, 'ARRIVAL_DELAY')
        plt.title('Arrival Delay vs ' + a)
        plt.show()
    return()

#describe_attributes(phx_data)

def bin_phx(data_frame, bin_list):
    data_frame['DEPARTURE_BIN'] = np.digitize(data_frame['SCHEDULED_DEPARTURE'], bin_list)
    data_frame.loc[data_frame['DEPARTURE_BIN'] == 6, 'DEPARTURE_BIN'] = 1
    return(data_frame)

def cancel_to_num(data_frame):
    data_frame.loc[data_frame['CANCELLED'] == 1, 'ARRIVAL_DELAY'] = 31
    return(data_frame)
    
def neg_to_zero(data_frame):
    data_frame.loc[data_frame['ARRIVAL_DELAY'] < 0, 'ARRIVAL_DELAY'] = 0
    return(data_frame)

def log_variable(data_frame):
    data_frame['LATE'] = np.nan
    data_frame.loc[data_frame['ARRIVAL_DELAY'] <= 30, 'LATE'] = 0
    data_frame.loc[data_frame['ARRIVAL_DELAY'] > 30, 'LATE'] = 1
    return(data_frame)

def data_prep(airport):
    
    flight_data = pd.read_csv(pathname_in)

    #phl_flight_data = flight_data[flight_data['ORIGIN_AIRPORT'] == 'PHL']
    #phl_flight_data.to_csv('/Users/madeline/Downloads/flight-delays/phl.csv')

    temp_flight_data = flight_data[flight_data['ORIGIN_AIRPORT'] == airport]
    temp_airline_list = temp_flight_data['AIRLINE'].unique().tolist()
    
    temp_flight_data = temp_flight_data.drop(columns=columns_to_drop)
    temp_bin_data = bin_phx(temp_flight_data, departure_bins)
    temp_bin_data = cancel_to_num(temp_bin_data)
    temp_bin_data = neg_to_zero(temp_bin_data)
    temp_bin_data = log_variable(temp_bin_data)
    n_obs = int(temp_bin_data.shape[0] * 0.1)
    print(n_obs)
    
    test_data = temp_bin_data.sample(n=n_obs, random_state=1)

    test_data.to_csv(airport+ '_test_data.csv')
    index_list = test_data.index.tolist()

    train_data = temp_bin_data[~temp_bin_data.index.isin(index_list)]

    train_data.to_csv(airport + '_train.csv')

    return(train_data)

phx_data = data_prep('PHX')

#what to bin: scheduled departure
#rid of months
#    
