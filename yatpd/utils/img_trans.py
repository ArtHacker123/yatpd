# -*- coding: utf-8 -*-

import cv2


def img_trans(img_data, trans_type):
    ''' Using various transformations to images.

    Parameters
    ----------
    img_data: np.ndarray
      data of image

    trans_type: str
      gray | LUV | Gabor
    '''
    if trans_type == 'gray':
        return (cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY), )
    elif trans_type == 'LUV':
        img_luv = cv2.cvtColor(img_data, cv2.COLOR_BGR2LUV)
        l_channel, u_channel, v_channel = cv2.split(img_luv)
        return l_channel, u_channel, v_channel
    elif trans_type == 'Gabor':
        img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        img_gabor_list = []
        for i in range(4):
            theta = np.pi * i / 4
            # Temporary Magic number
            gabor_kernel = cv2.getGaborKernel((25, 25), 5, theta, 50,
                                              5 / 25., ktype=cv2.CV_32F)
            img_gabor_list.append(cv2.filter2D(img_gray, cv2.CV_32F,
                                               gabor_kernel))
        return tuple(img_gabor_list)
    else:
        raise Exception("Type %s trans_type not support" % trans_type)
