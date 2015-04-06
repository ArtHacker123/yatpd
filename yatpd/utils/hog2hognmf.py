# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import NMF


def hog2hognmf(hog_feature):
    '''Transform HOG feature into HOG-NMF feature

    Parameters
    ----------
    hog_feature: np.ndarray
      HOG feature
    '''
    mat = np.zeros((500, 8), dtype=np.float32)
    NMFmodel = NMF(n_components=2, init='random', random_state=0)
    # Transform 3780 into 500 * 8
    for i in range(7):
        mat[:, i] = hog_feature[i * 500:(i + 1) * 500]
    mat[:280, 7] = hog_feature[3500:]
    W = NMFmodel.fit_transform(mat)
    H = NMFmodel.components_
    hognmf_feature = np.array([], dtype=np.float32)
    for i in range(8):
        W[:, i] /= np.sum(W[:, i])
        hognmf_feature = np.append(hognmf_feature,
                                   W[:, i])
    for i in range(500):
        H[i, :] /= np.sum(H[i, :])
        hognmf_feature = np.append(hognmf_feature,
                                   H[i, :])
    return hognmf_feature
