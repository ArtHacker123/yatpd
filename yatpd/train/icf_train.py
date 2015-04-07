# -*- coding: utf-8 -*-

import cv2
import numpy as np
from ..utils import timer
from ..utils import img_trans
from ..utils import hog2hognmf
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier


@timer
def icf_train(img_data_list, channel_type, feature_type, classifier):
    ''' Use ICF to train a model.
    NOTE: the size of image should be same.

    Parameters
    ----------
    img_data_list: list of tuple
      img_data_list is a list that consists of tuple like (img_data, flag).
      For flag, 1 stand for positive data, -1 stand for negative data.

    channel_type: str
      Gray | LUV | Gabor | DoG

    feature_type: str
      HOG | HOG-NMF

    classifier: str
      AdaBoost | SVM
    '''
    img_size = img_data_list[0][0].shape
    img_feature_list = []
    img_flag_list = []
    for img_data in img_data_list:
        channel_list = img_trans(img_data[0], channel_type)
        img_feature = np.array([], dtype=np.float32)
        hog = cv2.HOGDescriptor()
        for channel in channel_list:
            if feature_type == 'HOG' or feature_type == 'HOG-NMF':
                hog_feature = hog.compute(channel)
                if feature_type == 'HOG':
                    img_feature = np.append(img_feature, hog_feature[:, 0])
                else:
                    img_feature = np.append(img_feature,
                                            hog2hognmf(hog_feature[:, 0]))
        img_feature_list.append(img_feature)
        img_flag_list.append(img_data[1])
    img_flag_list = np.array(img_flag_list, dtype=np.int32)
    img_feature_list = np.array(img_feature_list, dtype=np.float32)
    if classifier == 'AdaBoost':
        boost_model = AdaBoostClassifier()
        boost_model.fit(img_feature_list, img_flag_list)
        return boost, img_size
    elif classifier == 'SVM':
        svm_model = svm.SVC(kernel='rbf')
        svm_model.fit(img_feature_list, img_flag_list)
        return svm_model, img_size
    else:
        raise Exception('Classifier doesn\'t support %s' % classifier)
