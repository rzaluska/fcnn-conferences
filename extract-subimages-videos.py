# script for extracting patches from video frames suitable for neural network
# training

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from PIL import Image
import sys
import os
import glob
from PIL import Image

from os.path import basename, splitext

import numpy as np

def acceptable(a):
    if np.average(a) > 0.95 * 255:
        return False
    return True


overlap = 8

source_path = './reducted-conferences-videos-equations/'
destination_path = './conferences-videos-equations-samples-512/'

for file_name in glob.glob(source_path+ "*.jpg"):
    name_without_extenstion = splitext(basename(file_name))[0]
    gt_file_path = source_path + name_without_extenstion + ".gt.jpg"
    print(file_name)
    print(gt_file_path)

    source_image = load_img(file_name, grayscale=False)
    try:
        groud_img = load_img(gt_file_path, grayscale=True)
    except FileNotFoundError:
        #groud_img = Image.new('RGB', (source_image.size[0], source_image.size[1]), (255, 255, 255))
        continue

    size_list = [512]

    for size_x in size_list:
        for size_y in size_list:
            subimage_size = (size_x, size_y)
            num_of_subimages_horizontal = source_image.size[0] // (subimage_size[0] // overlap)
            num_of_subimages_vertical = source_image.size[1] // (subimage_size[1] // overlap)
            rest_h = source_image.size[0] - num_of_subimages_horizontal * (subimage_size[0] // overlap)
            rest_v = source_image.size[1] - num_of_subimages_vertical * (subimage_size[1] // overlap)
            for i in range(num_of_subimages_horizontal):
                for j in range(num_of_subimages_vertical):
                    x = i * (subimage_size[0] // overlap)
                    y = j * (subimage_size[1] // overlap)
                    w = x + (subimage_size[0])
                    h = y + (subimage_size[1])
                    crop_rect = (x,y,w,h)
                    if w > source_image.size[0] or h > source_image.size[1]:
                        continue
                    chunk_file_name = "{dir}{name}-{sizex}-{sizey}-{i}-{j}".format(dir=destination_path, i=i, j=j, name=name_without_extenstion, sizex=size_x, sizey=size_y)
                    gt_sub_image = groud_img.crop(crop_rect)
                    if not acceptable(img_to_array(gt_sub_image)):
                        continue
                    print(chunk_file_name)
                    gt_sub_image.save(chunk_file_name + ".gt.jpg")
                    sub_image = source_image.crop(crop_rect)
                    sub_image.save(chunk_file_name+ ".jpg")


