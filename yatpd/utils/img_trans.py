# -*- coding: utf-8 -*-

import cv2


def img_trans(img_data, trans_type):
    ''' Using various transformations to images.

    Parameters
    ----------
    img_data: np.ndarray
      data of image

    trans_type: str
      Gray | LUV | Gabor | DoG
    '''
    if trans_type == 'Gray':
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
    elif trans_type == 'DoG':
        img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        # Temporary Magic number
        img_gaussian_1 = cv.GaussianBlur(img_gray, (1, 1), 0)
        img_gaussian_3 = cv.GaussianBlur(img_gray, (3, 3), 0)
        img_gaussian_5 = cv.GaussianBlur(img_gray, (5, 5), 0)
        return (img_gaussian_3 - img_gaussian_1,
                img_gaussian_5 - img_gaussian_3)
    else:
        raise Exception("Trans_type %s trans_type not support" % trans_type)
