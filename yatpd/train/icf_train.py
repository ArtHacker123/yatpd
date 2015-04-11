# -*- coding: utf-8 -*-

from ..utils import get_icf_feature
from sklearn.ensemble import AdaBoostClassifier


def icf_train(img_data_list, n_estimators=1000):
    ''' Train a ICF model.

    Parameters
    ----------
    img_data_list: list of tuple
      img_data_list is a list that consists of tuple like (img_data, flag).
      For flag, 1 stand for positive data, -1 stand for negative data.

    n_estimators: int
      n_estimators in AdaBoostClassifier
      Defualt n_estimators is 1000
    '''
    raw_img_data_list, img_flag_list = [], []
    for raw_img_data, flag in img_data_list:
        raw_img_data_list.append(raw_img_data)
        img_flag_list.append(flag)
    img_feature_list, feature_config = get_icf_feature(img_raw_data_list)
    img_feature_list = np.array(img_feature_list, ntype=np.float32)
    img_flag_list = np.array(img_flag_list, ntype=np.int32)
    boost_model = AdaBoostClassifier(n_estimators=n_estimators)
    boost_model.fit(img_feature_list, img_flag_list)
    return boost_model, feature_config
