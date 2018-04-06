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

def print_err(*args, **kwargs):
    print(*args, file=stderr, **kwargs)

def gate(i):
    if i < 0.2:
        return 0
    else:
        return 1

model = keras.models.load_model('conferences-fcnn.h5')

f = np.vectorize(gate)

for file_path in sorted(glob.glob("frames2/*.jpg")):
	print(file_path)
	file_name = os.path.splitext(os.path.basename(file_path))[0]
	img = load_img(file_path, grayscale=False)
	a = img_to_array(img) / 255.0
	predictions = model.predict(numpy.array([a]))[0]
	predictions = f(predictions)
	predictions = predictions * 255
	output_image = array_to_img(predictions)
	#crop_rect = (0,0,img.size[0],img.size[1])
	#output_image.paste(pil_image, crop_rect)
	#output_image.save("predicted_frames/"+ file_name + ".png")
	postprocess_and_save(output_image, img, "predicted_frames2/"+ file_name + ".jpg")

