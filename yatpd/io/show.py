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
      image data
    '''
    cv2.show(window_tile, img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
