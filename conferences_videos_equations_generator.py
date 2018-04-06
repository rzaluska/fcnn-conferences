# simple batch generator

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from PIL import Image
import sys
import os
import glob
from random import shuffle
from os.path import basename, splitext
import numpy as np

def ConferencesBatchGenerator(source_dir, size):
    x_data = []
    y_data = []

    images_path = './' + source_dir + "/"
    list_of_images = list(glob.glob(images_path + "*.gt.jpg"))
    shuffle(list_of_images)

    for file_name in list_of_images:
        name_wihout_extension = splitext(basename(file_name))[0][:-3]
        #print(name_wihout_extension)
        x_image  = images_path + name_wihout_extension + ".jpg"
        y_image  = images_path + name_wihout_extension + ".gt.jpg"

        source_image = load_img(x_image, grayscale=False)
        ground_img = load_img(y_image, grayscale=True)
        a = img_to_array(source_image) / 255.0
        x_data.append(a)
        a = img_to_array(ground_img) / 255.0
        y_data.append(a)

        if len(x_data) == size:
            x = np.array(x_data)
            y = np.array(y_data)
            x_data = []
            y_data = []
            yield x,y
