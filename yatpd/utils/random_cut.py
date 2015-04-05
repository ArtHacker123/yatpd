# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from random import randint


def random_cut(img_data, cut_size):
    '''Cut image in random position

    Parameters
    ----------
    img_data: np.ndarray
      data of image

    cut_size: tuple
      (height, width)
    '''
    img_size = img_data.shape
    if img_size[0] < cut_size[0] or img_size[1] < cut_size[1]:
        return None
    lf_x_begin = randint(0, img_size[0] - cut_size[0])
    lf_y_begin = randint(0, img_size[1] - cut_size[1])
    lf_x_end = lf_x_begin + cut_size[0]
    lf_y_end = lf_y_begin + cut_size[1]
    return img_data[lf_x_begin:lf_x_end, lf_y_begin:lf_y_end, :]


def random_cut_list(img_data_list, cut_size):
    '''Cut image in random position

    Parameters
    ----------
    img_data_list: list
      list of image_data

    cut_size: tuple
      (height, width)
    '''
    ret_img_data_list = []
    for img_data in img_data_list:
        ret_img_data = random_cut(img_data, cut_size)
        if ret_img_data is None:
            continue
        ret_img_data_list.append(ret_img_data)
    return ret_img_data_list
