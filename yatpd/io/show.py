# -*- coding: utf-8 -*-

import cv2
import numpy as np


def imshow(window_title, img_data):
    '''Show a single image.

    Parameters
    ----------
    window_title: str
      Title of the window.

    img_data: numpy.ndarray
      Data of image.
    '''
    cv2.imshow(window_title, img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
