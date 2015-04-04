# -*- coding: utf-8 -*-

import cv2


def draw(img_data, lt, rb, color, line_width):
    '''Draw a rectangle.

    Parameters
    ----------
    img_data: np.narray
      data of image
    lt: tuple
      left top point of rectangle
    rb: tuple
      right bottom point of rectangle
    color: tuple | str
      red | green | blue
      (R, G, B)
    line_width: int
      width of line
    '''
    color_dict = {'red': (255, 0, 0),
                  'green': (0, 255, 0),
                  'blue': (0, 0, 255)}
    if type(color) == str:
        color = color_dict[color]
    cv2.rectangle(img_data, lt, rb,
                  color, line_width)
