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

'''
SGD CLASSIFIER
'''
# CROSS VALIDATION 
penalty = ['l1', 'l2']
loss = ['hinge', 'log']
epsilon = [0.01, 0.1, 0.5]

sgd_params = {'penalty': penalty,
             'loss': loss,
             'epsilon': epsilon}

# Time executions
import time
start_time = time.time()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images.astype(np.float64))
test_images_scaled = scaler.fit_transform(test_images.astype(np.float64))

from sklearn.linear_model import SGDClassifier

sgd_class = SGDClassifier(penalty='l2', loss='hinge', epsilon=0.1)
sgd_class.fit(train_images_scaled, train_labels)

sgd_predictions = sgd_class.predict(test_images_scaled)

print("SGD accuracy: ",accuracy_score(test_labels, sgd_predictions))
print("Time of execution: %.2f seconds" %(time.time()-start_time))

print()


