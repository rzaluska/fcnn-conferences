# train model on additional samples

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda, UpSampling2D, Deconvolution2D
from keras import backend as K
import keras
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

model = keras.models.load_model('conferences-fcnn.h5')

batch_size = 4

for e in range(1):
    print("=======epoch %d" % e)
    for X_train, Y_train in ConferencesBatchGenerator('conferences-videos-equations-samples-256', batch_size):
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
        model.save('conferences-fcnn.h5')
