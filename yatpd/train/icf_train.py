# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ..utils import timer
from ..utils import img_trans


@timer
def icf_train(img_data_list, channel_type, classifier):
    ''' Use ICF to train a model.
    NOTE: the size of image should be same.

    Parameters
    ----------
    img_data_list: list of tuple
      img_data_list is a list that consists of tuple like (img_data, flag).
      For flag, 1 stand for positive data, -1 stand for negative data.

    channel_type: str
      Gray | LUV | Gabor | DoG

    classifier: str
      boost | svm
    '''
    img_size = img_data_list[0][0].shape
    img_feature_list = []
    img_flag_list = []
    for img_data in img_data_list:
        channel_list = img_trans(img_data[0], channel_type)
        img_feature = np.array([], dtype=np.float32)
        for channel in channel_list:
            hog = cv2.HOGDescriptor()
            hog_feature = hog.compute(channel)
            img_feature = np.append(img_feature, hog_feature[:, 0])
        img_feature_list.append(img_feature)
        img_flag_list.append(img_data[1])
    if classifier == 'boost':
        img_flag_list = np.array(img_flag_list, dtype=np.float32)
        img_feature_list = np.array(img_feature_list, dtype=np.float32)
        boost = cv2.Boost()
        ret_flag = boost.train(img_feature_list, cv2.CV_ROW_SAMPLE,
                               img_flag_list)
        assert ret_flag is False
        return boost, img_size
    elif classifier == 'svm':
        img_flag_list = np.array(img_flag_list, dtype=np.float32)
        img_feature_list = np.array(img_feature_list, dtype=np.float32)
        params = dict(kernel_type=cv2.SVM_RBF,
                      svm_type=cv2.SVM_C_SVC,
                      C=1,
                      gamma=0.5)
        svm = cv2.SVM()
        svm.train(img_feature_list, img_flag_list, params=params)
        return svm, img_size
    else:
        raise Exception('Classifier doesn\'t support %s' % classifier)
