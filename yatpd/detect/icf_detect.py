# -*- coding: utf-8 -*-

import cv2
import numpy as np
import ..utils import timer
import ..utils import img_trans
import ..utils import draw


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
      gray | LUV | Gabor | DoG
      same as channel_type of the models
    model_size: tuple
      Size of training data
    '''
    ret_img = np.array(img_data, copy=True)
    lower_img_data = cv2.pyrDown(img_data)
    higher_img_data = cv2.cv2.pyrUp(img_data)
    img_data_list = [(img_trans(lower_img_data, channel_type), 2, 'red'),
                     (img_trans(img_data, channel_type), 1, 'blue'),
                     (img_trans(higher_img_data, channel_type), 0.5, 'green')]
    img_size = img_size.shape
    for img_data, rate, color in img_data_list:
        for x in range(0, img_data[0] - model_size[0], 4):
            for y in range(0, img_data[1] - model_size[1], 4):
                img_feature = np.array()
                for channel in channel_list:
                    hog = cv2.HOGDescriptor()
                    hog_feature = hog.compute(channel)
                    np.append(img_feature, hog_feature)
                flag = model.predict(img_feature)
                if flag == 1:
                    def pos_trans(x):
                        return x * rate
                    draw(ret_img, (pos_trans(x), pos_trans(y)),
                         (pos_trans(x + model_size[0]),
                          pos_trans(y + model_size[1])),
                         color, 2)
    return ret_img
