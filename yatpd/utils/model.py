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


def load_model(model_type, file_path):
    '''Load model from a file.

    Parameters
    ----------
    model_type: str
      boost | svm

    file_path: str
      the path of file
    '''
    model_create_dict = {'boost': cv2.Boost,
                         'svm': cv2.SVM}
    model = model_create_dict[model_type]()
    model.load(file_path)
    return model
