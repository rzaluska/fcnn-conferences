# generate network raw prediction

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

from sys import stderr

def print_err(*args, **kwargs):
    print(*args, file=stderr, **kwargs)

def gate(i):
    if i < 0.2:
        return 0
    else:
        return 1

model = keras.models.load_model('conferences-fcnn.h5')

f = np.vectorize(gate)

file_path = sys.argv[1]
file_name = os.path.splitext(os.path.basename(file_path))[0]
img = load_img(file_path, grayscale=False)
#output_image = img.copy()
output_image = Image.new('RGBA', img.size)
alpha = Image.new('RGBA', img.size, (0, 0, 0, 128))
a = img_to_array(img) / 255.0
predictions = model.predict(numpy.array([a]))[0]
predictions = f(predictions)
predictions = predictions * 255
pil_image = array_to_img(predictions)
crop_rect = (0,0,img.size[0],img.size[1])
#output_image.paste(pil_image, crop_rect, alpha)
output_image.paste(pil_image, crop_rect)

output = BytesIO()
format = 'PNG'
output_image.save(output, format)
contents = output.getvalue()
os.write(sys.stdout.fileno(), contents)
output.close()
