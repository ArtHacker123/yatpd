# -*- coding: utf-8 -*-

from sklearn.externals import joblib


def save_model(model, file_path):
    '''Save model to a file.

    Parameters
    ----------
    model: object
      model

    file_path: str
      the path of file
    '''
    joblib.dump(model, file_path)


def load_model(file_path):
    '''Load model from a file.

    Parameters
    ----------
    file_path: str
      the path of file
    '''
    model = joblib.load(file_path)
    return model
