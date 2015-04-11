# -*- coding: utf-8 -*-

import cv2
import os


def imsave_list(img_data_list, folder_path):
    '''Save images into a folder.

    Parameters
    ----------
    img_data_list: list
      List of img_data.

    folder_path: str
      Path of folder.
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    cnt = 0
    for img_data in img_data_list:
        file_path = os.path.join(folder_path, str(cnt) + '.jpg')
        cv2.imwrite(file_path, img_data)
        cnt += 1
