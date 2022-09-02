from typing import List, Tuple
import utils.file_parser as fp
from utils.function_tracer import FunctionTracer
from utils.image import (
    StrideImage,
    to_image,
    to_stride_image,
)
from einops import rearrange
import numpy as np
from scipy.signal import convolve2d
from datetime import datetime
import argparse


def convert_to_numpy(stride_image):
    red = rearrange(np.array(stride_image.pixels_red, dtype='int64'), '(h w) -> h w', 
                    w=stride_image.resolution.width, 
                    h=stride_image.resolution.height)
    return red

def create_f1_mask(size):
    '''
     -----     
    |*****|
    |*   *|
    |* * *|
    |*   *|
    |*****|
     -----
    '''
    mask = np.zeros((size*5, size*5), dtype='int')
    mask[:,:size] = 1
    mask[:,-size:] = 1
    mask[:size,:] = 1
    mask[-size:, :] = 1
    mask[size*2:size*3, size*2:size*3] = 1
    return mask

def create_f2_mask(size):
    '''
     -----     
    |     |
    |     |
    | * * |
    |     |
    |     |
     -----
    '''
    mask = np.zeros((size*5, size*5), dtype='int')
    mask[size*2:size*3,size:size*2] = 1
    mask[size*2:size*3,size*3:size*4] = 1
    return mask

def create_f3_mask(size):
    '''
     -----     
    |     |
    |  *  |
    |     |
    |  *  |
    |     |
     -----
    '''
    mask = np.zeros((size*5, size*5), dtype='int')
    mask[size:size*2, size*2:size*3] = 1
    mask[size*3:size*4, size*2:size*3] = 1
    return mask

def create_f4_mask(size):
    '''
     -----     
    |     |
    | * * |
    |     |
    | * * |
    |     |
     -----
    '''
    mask = np.zeros((size*5, size*5), dtype='int')
    mask[size*1:size*2,size*1:size*2] = 1
    mask[size*1:size*2,size*3:size*4] = 1
    mask[size*3:size*4,size*1:size*2] = 1
    mask[size*3:size*4,size*3:size*4] = 1
    return mask

def correct_one_eye(image, threshold_image, size, x, y):
    if (threshold_image[x-size*4:x+1, y-size*4:y+1]*f1_masks[size]).sum() == 17*size**2:
        image[x-size*4:x+1, y-size*4:y+1] = image[x-size*4:x+1, y-size*4:y+1] - f1_masks[size]*150
        
    if (threshold_image[x-size*4:x+1, y-size*4:y+1]*f2_masks[size]).sum() == 2*size**2:
        image[x-size*4:x+1, y-size*4:y+1] = image[x-size*4:x+1, y-size*4:y+1] - f2_masks[size]*150

    if (threshold_image[x-size*4:x+1, y-size*4:y+1]*f3_masks[size]).sum() == 2*size**2:
        image[x-size*4:x+1, y-size*4:y+1] = image[x-size*4:x+1, y-size*4:y+1] - f3_masks[size]*150

    if (threshold_image[x-size*4:x+1, y-size*4:y+1]*f4_masks[size]).sum() == 4*size**2:
        image[x-size*4:x+1, y-size*4:y+1] = image[x-size*4:x+1, y-size*4:y+1] - f4_masks[size]*150
    return image

def remove_eyes(image_in):
    threshold_image = (image_in >= 200).astype('int')

    # go through masks in descending order:
    for size in sorted(f1_masks)[::-1]:
        mask = f1_masks[size]
        corrected_image = image_in[:]
        detected = convolve2d(threshold_image, mask) >= (17*size**2)
        if detected.sum() > 0:
            for x, y in zip(*np.where(detected)):
                corrected_image = correct_one_eye(corrected_image, threshold_image, size, x, y)
    return corrected_image

def unparse_pixel(rgba: Tuple[int]) -> int:
    encoded = sum([v*2**(a*8) for a,v in enumerate(rgba[::-1])])
    # if overflow:
    if encoded < 0: encoded = 4294967296 + encoded
    return encoded

def dump_images(images, output_file):
    ft = FunctionTracer("save images to bin file", "seconds")
    with open(output_file, 'w') as wfile:
        wfile.write(str(len(images)) + '\n')
        for image in images:
            wfile.write(f'{image.resolution.width} {image.resolution.height}\n')
            pixels = image.merge_pixel_components()
            pixels = [str(unparse_pixel((p.red, p.green, p.blue, p.alpha))) for p in pixels]
            wfile.write(' '.join(pixels) + '\n')
    del ft

def compute_solution(input_images):
    ft = FunctionTracer("compute_solution", "seconds")
    cleared_images = []
    for image_in in input_images:
        np_image_in = convert_to_numpy(image_in)
        corrected_red_channel = remove_eyes(np_image_in)
        image_in.pixels_red = list(corrected_red_channel.flatten())
        cleared_images.append(image_in)
    del ft
    return cleared_images

f1_masks = {size: create_f1_mask(size) for size in range(1, 2)}
f2_masks = {size: create_f2_mask(size) for size in range(1, 2)}
f3_masks = {size: create_f3_mask(size) for size in range(1, 2)}
f4_masks = {size: create_f4_mask(size) for size in range(1, 2)}

