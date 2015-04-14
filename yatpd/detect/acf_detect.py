# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ..utils import get_acf_feature
from ..utils import timer
from ..utils import draw


@timer
def acf_detect(model, img_data, model_size):
    ''' Detect a single image by using ACF-like model.

    Parameters
    ----------
    model: Object
      ACF-like model.

    img_data: np.ndarray
      Data of image.

    model_size: tuple
      Size of training data.
    '''
    ret_img = np.array(img_data, copy=True)
    lower_img_data = cv2.pyrDown(img_data)
    higher_img_data = cv2.pyrUp(img_data)
    img_data_list = [(lower_img_data, 2, 'red'),
                     (img_data, 1, 'blue'),
                     (higher_img_data, 0.5, 'green')]
    for img_data, rate, color in img_data_list[:2]:
        img_size = img_data.shape
        pos_list = []
        img_feature_list = []
        for x in range(0, img_size[0] - model_size[0], 8):
            for y in range(0, img_size[1] - model_size[1], 8):
                pos_list.append((x, y))
                img_slice = img_data[x:x + model_size[0],
                                     y:y + model_size[1]]
                img_feature = get_acf_feature([img_slice])
                img_feature_list += img_feature
        img_feature_list = np.array(img_feature_list, dtype=np.float32)
        result_list = model.predict(img_feature_list)
        for flag, pos in zip(result_list, pos_list):
            def pos_trans(x):
                return int(x * rate)
            if flag > 0:
                x, y = pos
                draw(ret_img, (pos_trans(x), pos_trans(y)),
                     (pos_trans(x + model_size[0]),
                      pos_trans(y + model_size[1])),
                     color, 2)
    return ret_img
