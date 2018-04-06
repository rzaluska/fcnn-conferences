import glob
import sys
from postprocess_fcnn_segmentation import postprocess_and_save
from io import BytesIO
from PIL import Image, ImageDraw, ImageOps
import sys
import os
import numpy
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
import colorsys
import glob
from sys import stderr
from postprocess_fcnn_segmentation import postprocess_and_save

raw_p = load_img(sys.argv[1], grayscale=True)
img = load_img(sys.argv[2], grayscale=False)
postprocess_and_save(raw_p, img, sys.argv[3])
