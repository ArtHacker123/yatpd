# -*- coding: utf-8 -*-

import cv2
from .img_trans import img_trans


def hog2mats(img_data):
    '''Get HOG(6 bins, 6 channels) in matrix.

    Parameters
    ----------
    img_data: np.ndarray
      data of image
    '''
    hog = cv2.HOGDescriptor(_winSize=(64, 128),
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=6)
    hog_feature = hog.compute(img_data)
    hog_mat = []
    for shift in range(6):
        single_bin_list = hog_feature[shift::6]
        single_bin_mat = np.zeros((15, 7), ntype=np.float32)
        for row in range(15):
            single_bin_mat[row, :] = single_bin_list[row * 15:(row + 1) * 15]
        hog_mat.append(single_bin_mat)
    return hog_mat


def luv2mats(img_data):
    '''Get LUV(3 channels) in matrix.

    Parameters
    ----------
    img_data: np.ndarray
      data of image
    '''
    tuple_luv = img_trans(img_data, 'LUV')
    return [tuple_luv[0], tuple_luv[1], tuple_luv[2]]

    
def sobel2mat(img_data):
    '''Get Sobel(1 channel) in matrix.

    Parameters
    ----------
    img_data: np.ndarray
      data of image
    '''
    gray_img = img_data(img_data, 'Gray')[0]
    sobel_mats = cv2.Sobel(_, cv2.CV_8U, 1, 0)
    return [sobel_mats]


def get_icf_feature(img_data_list, feature_config=None):
    '''Get ICF feature.
       Sum of random rectangle in gradient histogram(6 bins), grad. and LUV.

    Parameters
    ----------
    img_data_list: list
      list of img_data

    feature_config: list or None
      position and size of rectangle feature.
      For None, function will generate a config list.
    '''
    feature_list = []
