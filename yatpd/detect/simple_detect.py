# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ..utils import timer
from ..utils import img_trans
from ..utils import draw
from ..utils import hog2hognmf
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier


@timer
def simple_detect(model, img_data, channel_type, feature_type, model_size):
    ''' Detect a single image by using simple model.

    Parameters
    ----------
    model: Object
      Simple model.

    img_data: np.ndarray
      Data of image.

    channel_type: str
      Gray | LUV | Gabor | DoG
      Same as channel_type of the models.

    feature_type: str
      HOG | HOG-NMF

    model_size: tuple
      Size of training data.
    '''
    ret_img = np.array(img_data, copy=True)
    lower_img_data = cv2.pyrDown(img_data)
    higher_img_data = cv2.pyrUp(img_data)
    img_data_list = [(lower_img_data, 2, 'red'),
                     (img_data, 1, 'blue'),
                     (higher_img_data, 0.5, 'green')]
    hog = cv2.HOGDescriptor()
    for img_data, rate, color in img_data_list[:2]:
        img_size = img_data.shape
        channel_list = img_trans(img_data, channel_type)
        pos_list = []
        feature_list = []
        for x in range(0, img_size[0] - model_size[0], 8):
            for y in range(0, img_size[1] - model_size[1], 8):
                pos_list.append((x, y))
                img_feature = np.array([], dtype=np.float32)
                for channel in channel_list:
                    channel_slice = channel[x:x + model_size[0],
                                            y:y + model_size[1]]
                    if feature_type == 'HOG' or feature_type == 'HOG-NMF':
                        hog_feature = hog.compute(channel_slice)
                        if feature_type == 'HOG':
                            img_feature = np.append(img_feature,
                                                    hog_feature[:, 0])
                        else:
                            img_feature = np.append(img_feature,
                                                    hog2hognmf(hog_feature[:,
                                                                           0]))
                feature_list.append(img_feature)
        feature_list = np.array(feature_list, dtype=np.float32)
        result_list = model.predict(feature_list)
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
