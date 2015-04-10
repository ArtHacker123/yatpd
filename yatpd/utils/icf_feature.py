# -*- coding: utf-8 -*-

import cv2


def hog2mat(img_data):
    '''Get HOG(6 bins) in matrix.

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
