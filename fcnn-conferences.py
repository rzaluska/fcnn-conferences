# neural network training script

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda, UpSampling2D, Deconvolution2D
from keras import backend as K
from keras.backend import tf as ktf
import numpy as np
from PIL import Image
from conferences_videos_equations_generator import ConferencesBatchGenerator

def converter(x):

    #x has shape (batch, width, height, channels)
    return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])

num_channels = 3

if K.image_data_format() == 'channels_first':
    input_shape = (num_channels, None, None)
else:
    input_shape = (None, None, num_channels)

model = Sequential()

model.add(Conv2D(512, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(UpSampling2D(size=(2,2)))
model.add(Deconvolution2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(UpSampling2D(size=(2,2)))
model.add(Deconvolution2D(128, (3, 3)))
model.add(Activation('relu'))

model.add(UpSampling2D(size=(2,2)))
model.add(Deconvolution2D(256, (3, 3)))
model.add(Activation('relu'))

model.add(UpSampling2D(size=(2,2)))
model.add(Deconvolution2D(512, (3, 3)))
model.add(Activation('relu'))

#model.add(Lambda(converter))

model.add(Deconvolution2D(1, (3, 3)))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              #optimizer='adam',
              metrics=['accuracy'])

batch_size = 16
for e in range(1):
    for X_train, Y_train in ConferencesBatchGenerator('conferences-videos-equations-samples-128', batch_size):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
        model.save('conferences-fcnn.h5')

batch_size = 4
for e in range(1):
    for X_train, Y_train in ConferencesBatchGenerator('conferences-videos-equations-samples-256', batch_size):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
        model.save('conferences-fcnn.h5')

batch_size = 1
for e in range(1):
    for X_train, Y_train in ConferencesBatchGenerator('conferences-videos-equations-samples-512', batch_size):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
        model.save('conferences-fcnn.h5')
