#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:08:28 2018

@author: micheles
"""

################################################################################################################
## Imports 

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics import r2_score


################################################################################################################
## Paths and Constants

np.random.seed(44234)




################################################################################################################
## Main

# Simulating the data
N = 10000
m_features = 40000

all_features = []
for i in range(m_features):
    x = np.random.normal(size = N).astype(np.float32).reshape((N, 1))
    all_features.append(x)

X = np.concatenate((all_features), axis = 1)            # X.shape: (N, m_features)

epsilon = np.random.normal(size = N, scale = 7.5)
y = 1.1 + 3.2 * X[:, 0] + epsilon
y = y.astype(np.float32).reshape((N, 1))                # y.shape: (N, 1)


## Display data
plt.scatter(X[:, 0], y[:, 0])
plt.title("y vs x_1: positive relationship")
plt.show()

plt.scatter(X[:, 1], y[:, 0])
plt.title("y vs x_2: no relationship")
plt.show()

plt.scatter(X[:, 2], y[:, 0])
plt.title("y vs x_3: no relationship")
plt.show()




######################################
# Sklearn approach

from sklearn.linear_model import Ridge

start = time.time()

# Training
clf = Ridge(alpha=1.0)
log_train = clf.fit(X, y)

b = clf.coef_[0]

print("%.2f sec."%(time.time() - start), end=' - ')
print("Coefficients for x_1, x_2, x_3 are %.3f, %.3f, %.3f, respectively" %
          (b[0], b[1], b[2]))

# Testing
y_hat = clf.predict(X)
print("R square: %.3f"%r2_score(y,y_hat))           # R^2 (coefficient of determination) regression score function.



######################################
# xgboost approach

import xgboost as xgb

start = time.time()

#Fitting XGB regressor
clf = xgb.XGBRegressor()
log_fit = clf.fit(X,y)
print(log_fit)

print("%.2f sec."%(time.time() - start), end=' - ')

# Testing
y_hat = clf.predict(X)
print("R square: %.3f"%r2_score(y,y_hat))           # R^2 (coefficient of determination) regression score function.



######################################
# Keras approach

import keras
from keras.layers import Input, Dense
from keras.models import Model


#Define the model
def regressor_model(m_features, regularizer=None, learning_rate=0.01):
    
    input_x = Input(shape = (m_features,))
    lin_fn = Dense(1, activation = None, kernel_regularizer = regularizer)(input_x)
    yx_model = Model(inputs = input_x, outputs = lin_fn)
    sgd = keras.optimizers.SGD(lr=learning_rate)
    yx_model.compile(loss = 'mean_squared_error', optimizer = sgd)
    
    return yx_model


reg_par = [0, 0.001, 0.01, 0.1, 1]


for i_reg in reg_par:
    start = time.time()
    
    # Training
    yx_model = regressor_model(m_features,learning_rate=0.1,regularizer=keras.regularizers.l2(i_reg))
    log_train = yx_model.fit(X, y, epochs = 50, batch_size = N, verbose = 0)

    weights = yx_model.get_weights()
    
    b = weights[0]
    
    print("%.2f sec."%(time.time() - start), end=' - ')
    print("Coefficients for x_1, x_2, x_3 are %.3f, %.3f, %.3f, respectively" %
          (b[0], b[1], b[2]))
    
    # Testing
    y_hat = yx_model.predict(X)
    print("R square: %.3f"%r2_score(y,y_hat))           # R^2 (coefficient of determination) regression score function.

# In this case, no regularization is needed!




# With many regularisation terms, you may need to insert at the end of every iteration:
#from keras import backend as K
#K.clear_session()

