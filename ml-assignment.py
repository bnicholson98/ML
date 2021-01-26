#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:01:35 2020

@author: ben
"""


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# reshape the dataset
nsamples, nx, ny = train_images.shape
train_images = train_images.reshape((nsamples,nx*ny))
nsamples, nx, ny = test_images.shape
test_images = test_images.reshape((nsamples,nx*ny))

np.random.seed(42)   # if you want reproducible results set the random seed value.
shuffle_index = np.random.permutation(60000)
train_images, train_labels = train_images[shuffle_index], train_labels[shuffle_index]


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

# Cross val
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
'''
RANDOM FOREST
'''
# =============================================================================
# # CROSS VALIDATION 
# n_estimators = [100, 200,300]
# max_features = ['log2','sqrt']
# bootstrap = [True,False]
# 
# rf_params = {'n_estimators': n_estimators,
#              'max_features': max_features,
#              'bootstrap': bootstrap}
# from sklearn.ensemble import RandomForestClassifier
# 
# rf_class = RandomForestClassifier() # 0.8827, 29 seconds
# 
# rf_cross_val = GridSearchCV(estimator=rf_class, param_grid=rf_params, cv=5, verbose=2, n_jobs=-1,  scoring='neg_mean_squared_error')
# 
# rf_cross_val.fit(train_images, train_labels)
# 
# print('Done cross val')
# print(rf_cross_val.best_params_)
# print(rf_cross_val.best_estimator_)
# 
# cvres = rf_cross_val.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)
# =============================================================================
    
# Time executions
import time
start_time = time.time()

from sklearn.ensemble import RandomForestClassifier

rf_class = RandomForestClassifier(bootstrap=False, max_features='sqrt', n_estimators=300)
rf_class.fit(train_images, train_labels)
# Predict
rf_predictions = rf_class.predict(test_images)

# Print result
print("Random forest accuracy: ",accuracy_score(test_labels, rf_predictions))
print("Time of execution: %.2f seconds" %(time.time()-start_time))

print()

