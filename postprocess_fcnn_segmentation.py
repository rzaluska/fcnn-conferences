# postprocessing logic

from io import BytesIO
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import sys
import os
import numpy
import numpy as np
import colorsys

black = 0
white = 255

def position_legal(position, width, height):
    if position[0] < 0:
        return False
    if position[1] < 0:
        return False

    if position[0] >= width:
        return False
    if position[1] >= height:
        return False

    return True


def flood_fill(start_x, start_y, pixels, discovered, width, height):
    max_x = 0
    max_y = 0
    min_x = 9999
    min_y = 9999

    stack = []
    stack.append((start_x, start_y))

    while stack:
        position = stack.pop()

        if not position_legal(position, width, height):
            continue

        if pixels[position] == black and not discovered[position]:
            discovered[position] = 1
            if position[0] < min_x:
                min_x = position[0]
            if position[0] > max_x:
                max_x = position[0]

            if position[1] < min_y:
                min_y = position[1]
            if position[1] > max_y:
                max_y = position[1]


            stack.append((position[0] + 1, position[1]))
            stack.append((position[0] - 1, position[1]))
            stack.append((position[0], position[1] + 1))
            stack.append((position[0], position[1] - 1))

    return (min_x, min_y, max_x, max_y)


# Find all blobs in image and returs bounding boxes
def find_blobs(pil_image):
    boxes = []
    pixels = pil_image.load()
    width, height = pil_image.size
    discovered = np.zeros(pil_image.size)

    for i in range(width):
        for j in range(height):
            if discovered[i,j] == 0:
                if pixels[i,j] == black:
                    box = flood_fill(i,j, pixels, discovered, width, height)
                    boxes.append(box)
    return boxes


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image, 'RGBA')
    for box in boxes:
        draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), outline=(255,0,0,255))
        draw.rectangle(box, outline=(255,0,0,255))
        draw.rectangle((box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1), outline=(255,0,0,255), fill=(255,0,0,128))

    del draw


def local_contrast(crop_image):
    image = crop_image.convert('L')
    np_array = np.array(image)
    return np.mean(np_array)


def box_good(box, net_ans, base_img):
    width, height = (box[2] - box[0]), (box[3] - box[1])
    area = width * height
    if area < 2000 :
        return False

    if height > width:
        return False

    if 2*height > width:
        return False

    crop_image = base_img.crop(box)
    crop_net_ans = net_ans.crop(box)
    image = crop_image.convert('L')
    edges_image = image.filter(ImageFilter.FIND_EDGES)
    image_array = np.array(image)
    net_ans_array = np.array(crop_net_ans)
    edges_image_array = np.array(edges_image)

    if np.mean(net_ans_array) / 256.0 > 0.6:
       return False

    contrast = np.amax(image_array) - np.amin(image_array)
    if contrast < 50:
        return False

    local_contrast = np.mean(edges_image_array)
    if local_contrast < 13:
        return False

    return True

def filter_boxes(boxes, net_ans, base_img):
    new_boxes = []

    for box in boxes:
        if box_good(box, net_ans, base_img):
            new_boxes.append(box)
    print(len(new_boxes))
    return new_boxes

def postprocess_and_save(net_ans, base_img, out_img):
    boxes = find_blobs(net_ans)
    print("Without postprocessing: " + str(len(boxes)))
    boxes = filter_boxes(boxes, net_ans, base_img)
    print("After postprocessing: " + str(len(boxes)))
    draw_boxes(base_img, boxes)
    base_img.save(out_img)
