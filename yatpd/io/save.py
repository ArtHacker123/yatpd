# -*- coding: utf-8 -*-

import cv


def imsave_list(img_data_list, folder_path):
    '''Save images into a folder

    Parameters
    ----------
    img_data_list: list
      list of img_data

    folder_path: str
      path of folder
    '''
    if os.path.exists(floder_path):
        os.makedirs(floder_path)
    cnt = 0
    for i in img_data_list:
        file_path = os.path.join(floder_path, str(cnt) + '.jpg')
        cv.SaveImage(file_path, img_data)
