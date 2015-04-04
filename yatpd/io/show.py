# -*- coding: utf-8 -*-

import cv2
import numpy as np


def imshow(window_title, img_data):
    '''show a single image.

    Parameters
    ----------
    window_title: str
      title of the window

    img_data: numpy.ndarray
      data of image
    '''
    cv2.imshow(window_tile, img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
