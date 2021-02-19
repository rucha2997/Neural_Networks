# Gurav, Rucha
# 1001-773-732
# 2020_11_03
# Assignment-04-03
import pytest
import numpy as np
from cnn import CNN
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10, cifar100
import os


def test_train():
    
    cnn = CNN()
    classes = 10
    bs = 50
    epochs = 20
    train_samples = 800
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[0:train_samples,:]
    y_train = y_train[0:train_samples,:]
    x_test = x_test[0:train_samples,:]
    y_test = y_test[0:train_samples,:]
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    
    model = Sequential()  
    model.add(Conv2D(30, (1, 1), padding='same', activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, (2, 2), padding='same', activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(40, activation='sigmoid'))
    model.add(Dense(classes, activation='softmax'))
    o = keras.optimizers.RMSprop(learning_rate=0.002)
    model.compile(loss='categorical_crossentropy',optimizer=o,metrics=['accuracy'])
    model1 = model.fit(x_train, y_train,batch_size=bs,epochs=epochs)

    cnn.add_input_layer(shape=x_train.shape[1:], name="input")
    cnn.append_conv2d_layer(num_of_filters=30, kernel_size = 1, padding='same', strides=1, activation='sigmoid', name="c1")
    cnn.append_maxpooling2d_layer(pool_size=(2, 2), name='p1')
    cnn.append_conv2d_layer(num_of_filters=24, kernel_size = 3, padding='same', strides=1, activation='sigmoid', name="c2")
    cnn.append_maxpooling2d_layer(pool_size=(3, 3), name='p2')
    cnn.append_flatten_layer(name='flat')
    cnn.append_dense_layer(num_nodes=100, activation='sigmoid', trainable=True, name='d1')
    cnn.append_dense_layer(num_nodes=classes, activation='softmax', trainable=True, name='d2')
    cnn.model.compile(loss='categorical_crossentropy',optimizer=o,metrics=['accuracy'])
    model2 = cnn.model.fit(x_train, y_train,batch_size=bs,epochs=epochs)
    assert np.allclose(model1.history['accuracy'], model2.history['accuracy'], rtol=1e-1, atol=1e-1*6)

    
def test_evaluate():

    cnn = CNN()
    classes = 10
    bs = 50
    epochs = 50
    train_samples = 500
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[0:train_samples, :]
    y_train = y_train[0:train_samples, :]
    x_test = x_test[0:train_samples, :]
    y_test = y_test[0:train_samples, :]
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    model = keras.Sequential()  
    model.add(Conv2D(45, (1, 1), padding='same', activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), padding='same', activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(classes, activation='softmax'))
    o = keras.optimizers.RMSprop(learning_rate=0.002)
    model.compile(loss='categorical_crossentropy',optimizer=o,metrics=['accuracy'])
    model1 = model.fit(x_train, y_train,batch_size=bs,epochs=epochs)
    evaluate1 = model.evaluate(x_test, y_test) 
    
    cnn.add_input_layer(x_train.shape[1:], name="input")
    cnn.append_conv2d_layer(num_of_filters=32, kernel_size = 1, padding='same', strides=1, activation='sigmoid', name="c1")
    cnn.append_maxpooling2d_layer(pool_size=(2, 2), name='p1')
    cnn.append_conv2d_layer(num_of_filters=25, kernel_size = 3, padding='same', strides=1, activation='sigmoid', name="c2")
    cnn.append_maxpooling2d_layer(pool_size=(3, 3), name='p2')
    cnn.append_flatten_layer(name='flat')
    cnn.append_dense_layer(num_nodes=150, activation='sigmoid', trainable=True, name='d1')
    cnn.append_dense_layer(num_nodes=classes, activation='softmax', trainable=True, name='d2')
    cnn.model.compile(loss='categorical_crossentropy',optimizer=o,metrics=['accuracy'])
    model2 = cnn.model.fit(x_train, y_train,batch_size=bs,epochs=epochs)
    evaluate2 = cnn.model.evaluate(x_test, y_test)
    assert np.allclose(evaluate1, evaluate2, rtol=1e-1, atol=1e-1*6)