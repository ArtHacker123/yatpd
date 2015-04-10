# -*- coding: utf-8 -*-

import cv2
from random import randint
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
    assert img_data_list == []
    img_raw_feature_list = []
    size_list = []
    for img_data in img_data_list:
        # Pre-smoothing
        img_data_pre_sm = cv2.GaussianBlur(img_data, (3, 3), 0)
        img_raw_feature = hog2mats(img_data_pre_sm) + \
            luv2mats(img_data_pre_sm) + \
            sobel2mat(img_data_pre_sm)
        # Post-smoothing
        img_raw_feature = map(lambda data: cv2.GaussianBlur(data, (3, 3), 0),
                              img_raw_feature)
        # Get integral image
        img_raw_feature = map(cv2.integral,
                              img_raw_feature)
        img_raw_feature_list.append(img_raw_feature)
    size_list = map(lambda data: data.shape,
                    img_raw_feature_list[0])
    if not feature_config:
        cnt = 0
        feature_config = []
        while True:
            channel_num = randint(0, 9)
            x_upper_bound, y_upper_bound = size_list[channel_num]
            beg_pos = (randint(0, x_upper_bound - 1),
                       randint(0, y_upper_bound - 1))
            end_pos = (randint(beg_pos[0], x_upper_bound - 1),
                       randint(beg_pos[1], y_upper_bound - 1))
            if (end_pos[0] - beg_pos[0]) * (end_pos[1] - beg_pos[0]) < 25:
                continue
            feature_config.append((channel_num, beg_pos, end_pos))
            cnt += 1
            if cnt >= 5000:
                break
    img_raw_feature_list = []
    for img_raw_feature in img_raw_feature_list:
        img_feature = np.zeros(5000, dtype=np.float32)
        for index, config in enumerate(feature_config):
            ch_now = img_raw_feature_list[config[0]]
            img_feature[index] = ch_now[config[2][0], config[2][1]] - \
                ch_now[config[1][0], config[1][1]]
        img_feature_list.append(img_feature)
    return img_feature_list, feature_config
