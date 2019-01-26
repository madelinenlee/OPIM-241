#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:08:41 2019

@author: madeline
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import calinski_harabaz_score


#read data in
boston_housing_data = pd.read_csv('/Users/madeline/Desktop/SPRING_2019/OPIM_241/Quiz_1/boston_housing.csv')
attributes = boston_housing_data.columns.tolist()

boston_test_data = boston_housing_data.sample(n=50, random_state=1)

boston_test_data.to_csv('/Users/madeline/Desktop/SPRING_2019/OPIM_241/Quiz_1/boston_test_data.csv')
index_list = boston_test_data.index.tolist()

boston_training_data = boston_housing_data[~boston_housing_data.index.isin(index_list)]
boston_training_data.to_csv('/Users/madeline/Desktop/SPRING_2019/OPIM_241/Quiz_1/boston_training_data.csv')

plt.matshow(boston_housing_data.corr())


#run through descriptive statistics
for attribute in attributes:
    print(boston_training_data[attribute].describe())


#function to calculate number of rows with missing values and return all observations
#with missing values
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




boston_training_data['median_value_bin'] = np.nan
for i in range(0, len(boston_training_data.shape[0])):
    


#create histograms for each attribute of data
for item in attributes:
    boston_housing_data[item].hist()
    plt.title(item)
    plt.savefig(item+'_hist.png')
    plt.show()
    
    
    
#code below adapted from from Prof. Lisa Singh, CS 587 Fall 2018--------------
cluster_data_frame = pd.concat([boston_housing_data['prop_resid_land_over_25k'], 
                                boston_housing_data['on_charles'], 
                                boston_housing_data['avg_rooms'],
                                boston_housing_data['tax_per_10k'],
                                #boston_housing_data['prop_built_before_1940'],
                                boston_housing_data['crime_rate_per_cap'],
                                boston_housing_data['nitric_oxide_concent'],
                                boston_housing_data['accessible_to_highways'],
                                boston_housing_data['student_teacher_ratio'],
                                boston_housing_data['prop_blacks'],
                                boston_housing_data['prop_low_income'],
                                #boston_housing_data['prop_industrial']
                                ],
                                axis=1, keys=['prop_resid_land_over_25k',
                                              'on_charles', 'avg_rooms',
                                              'tax_per_10k', 
                                              'prop_built_before_1940',
                                              'crime_rate_per_cap',
                                              'nitric_oxide_concent',
                                              'accessible_to_highways',
                                              'student_teacher_ratio',
                                              'prop_blacks',
                                              'prop_low_income',
                                              'prop_industrial'
                                              ])
    
x = cluster_data_frame.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalized_bh_data = pd.DataFrame(x_scaled)
pprint(normalized_bh_data[:10])


# Create clusters (If you try 3 and then 20, you will see how different
# it looks when you attempt to fit the data.
k = 2
kmeans = KMeans(n_clusters=k)
cluster_labels = kmeans.fit_predict(normalized_bh_data)

# Determine if the clustering is good
silhouette_avg = silhouette_score(normalized_bh_data, cluster_labels)
c_h_avg = calinski_harabaz_score(normalized_bh_data, cluster_labels)
print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
print("For n_clusters =", k, "The average calinski_harabaz_score is :", c_h_avg)

centroids = kmeans.cluster_centers_
pprint(cluster_labels)
pprint(centroids)

#pprint(prediction)

# See how it fits data on different dimensions
print(pd.crosstab(cluster_labels, boston_housing_data['avg_rooms']))
print(pd.crosstab(cluster_labels, boston_housing_data['tax_per_10k']))
print(pd.crosstab(cluster_labels, boston_housing_data['nitric_oxide_concent']))

#####
# PCA
# Let's convert our high dimensional data to 2 dimensions
# using PCA
pca2D = decomposition.PCA(2)

# Turn the NY Times data into two columns with PCA
plot_columns = pca2D.fit_transform(normalized_bh_data)

# Plot using a scatter plot and shade by cluster label
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
plt.show()

#-----------------------------------------------------------------------------