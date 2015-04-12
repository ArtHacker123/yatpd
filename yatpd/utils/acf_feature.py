# -*- coding: utf-8 -*-

import cv2
import numpy as np
from math import ceil
from .icf_feature import hog2mats, luv2mats, sobel2mat


def acf_smooth(img_data):
    '''Smooth matrix using [1 2 1] / 4 filter.

    Parameters
    ----------
    img_data: np.ndarray
      Data of image.
    '''
    length, width = img_data.shape
    s_img_data = np.zeros((length, width), dtype=np.float32)
    for x in range(0, length):
        s_img_data[x, 0] = img_data[x, 0]
        s_img_data[x, width - 1] = img_data[x, width - 1]
        for y in range(1, width - 1):
            pre_s_img_data[x, y] = (img_data[x, y - 1] +
                                    img_data[x, y] * 2 +
                                    img_data[x, y + 1]) / 4.
    return s_img_data


def get_acf_sum(img_data):
    '''Get sum of 4 * 4 matrix in a channel.

    Parameters
    ----------
    img_data: np.ndarray
      Data of image.
    '''
    length, width = img_data.shape
    s_length, s_width = ceil(length / 4.), ceil(width / 4.)
    sum_data = np.zeros((s_length, s_width), dtype=np.float32)
    fix_img_data = np.zeros((s_length * 4, s_width * 4,
                            dtype=np.float32))
    for x in range(0, length):
        fix_img_data[x, :width] = img_data[x, :]
    integral_data = cv2.integral(fix_img_data)
    for x in range(0, length, 4):
        for y in range(0, width, 4):
            sum_data[x / 4, y / 4] = (integral_data[x + 4, y + 4] -
                                      integral_data[x, y])
    return sum_data


def get_acf_feature(img_feature_list):
    '''Get ACF-like feature.
       Derivative feature from gradient histogram(6 bins), grad. and LUV.

    Parameters
    ----------
    img_data_list: list
      List of img_data.
    '''
    assert img_data_list != []
    img_feature_list = []
    for img_data in img_data_list:
        # Pre-smoothing
        img_data = acf_smooth(img_data)
        channel_list = luv2mats(img_data) + sobel2mat(img_data)
        img_feature_mats_list = []
        for channel in channel_list:
            img_feature_mats_list.append(get_acf_sum(channel))
        img_feature_mats_list += hog2mats(img_data)
        # Post-smoothing
        img_feature = np.array([], dtype=np.float32)
        for img_feature_mats in img_feature_mats_list:
            img_feature_mats = acf_smooth(img_feature_mats)
            for i in range(img_feature_mats.shape[0]):
                img_feature = np.append(img_feature,
                                        img_feature_mats[i, :])
        img_feature_list.append(img_feature)
    return img_feature_list
