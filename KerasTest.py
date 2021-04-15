# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:29:20 2021

@author: Jennifer
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #將類別向量轉為二元類別矩陣
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    x_test = np.random.normal(x_test) #加躁聲
    x_train = x_train/255
    x_test = x_test/255
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    
    #定義結構
    model = Sequential()
    
    model.add(Dense(input_dim=28*28, units=500, activation='relu'))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    
    #train model
    model.fit(x_train, y_train, batch_size=100, epochs=20)
    
    result_train = model.evaluate(x_train, y_train)
    result_test = model.evaluate(x_test, y_test)
    print('Train ACC:', result_train[1])
    print('Test ACC:', result_test[1])