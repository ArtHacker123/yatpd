# -*- encoding: utf-8 -*-

import cv2
import os


def imread(file_path):
    '''Read a single image from a file.

    Parameters
    ----------
    file_path: str
      file_path
    '''
    return cv2.imread(file_path)


def flread(folder_path):
    '''Read images from a folder.

    Parameters
    ----------
    folder_path: str
      folder_path
    '''
    img_type = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe',
                'png', 'pbm', 'pgm', 'ppm', 'sr',
                'ras', 'tiff', 'tif', 'exr', 'jp2']
    file_name_list = os.listdir(folder_path)
    img_list = []
    for file_name in file_name_list:
        file_path = os.path.join(folder_path, filename)
        if lower(os.path.splitext(file_path)[1]) not in img_type:
            continue
        img_list.append(cv2.imread(file_path))
    return img_list
