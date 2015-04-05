# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ..utils import timer
from ..utils import img_trans
from ..utils import draw


@timer
def icf_detect(model, img_data, channel_type, model_size):
    ''' Detect a single image by using ICF model.

    Parameters
    ----------
    model: Object
      ICF model
    img_data: np.ndarray
      data of image
    channel_type: str
      Gray | LUV | Gabor | DoG
      same as channel_type of the models
    model_size: tuple
      Size of training data
    '''
    ret_img = np.array(img_data, copy=True)
    lower_img_data = cv2.pyrDown(img_data)
    higher_img_data = cv2.pyrUp(img_data)
    img_data_list = [(lower_img_data, 2, 'red'),
                     (img_data, 1, 'blue'),
                     (higher_img_data, 0.5, 'green')]
    for img_data, rate, color in img_data_list:
        img_size = img_data.shape
        channel_list = img_trans(img_data, channel_type)
        for x in range(0, img_size[0] - model_size[0], 16):
            for y in range(0, img_size[1] - model_size[1], 16):
                img_feature = np.array([], dtype=np.float32)
                for channel in channel_list:
                    hog = cv2.HOGDescriptor()
                    channel_slice = channel[x:x + model_size[0],
                                            y:y + model_size[1]]
                    hog_feature = hog.compute(channel_slice)
                    img_feature = np.append(img_feature, hog_feature[:, 0])
                flag = model.predict(img_feature)
                if flag == 1:
                    def pos_trans(x):
                        return x * rate
                    draw(ret_img, (pos_trans(x), pos_trans(y)),
                         (pos_trans(x + model_size[0]),
                          pos_trans(y + model_size[1])),
                         color, 2)
    return ret_img
