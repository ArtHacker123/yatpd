# -*- coding: utf-8 -*-

import numpy as np
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


def get_acf_feature(img_feature_list):
    '''Get ACF-like feature.
       Derivative feature from gradient histogram(6 bins), grad. and LUV.

    Parameters
    ----------
    img_data_list: list
      List of img_data.
    '''
    assert img_data_list != []
    for img_data in img_data_list:
        # Pre-smoothing
