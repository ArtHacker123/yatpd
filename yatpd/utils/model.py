# -*- coding: utf-8 -*-

import cv2


def save_model(model, file_path):
    '''Save model to a file.

    Parameters
    ----------
    model: object
      model

    file_path: str
      the path of file
    '''
    model.save(file_path)


def load_model(model, file_path):
    '''Load model from a file.

    Parameters
    ----------
    model: object
      model

    file_path: str
      the path of file
    '''
    model.load(file_path)
