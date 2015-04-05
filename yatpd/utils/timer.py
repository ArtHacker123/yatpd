# -*- coding: utf-8 -*-

import time


def timer(func):
    '''A simple decorator to count time.
    '''
    def wrapper(*args, **kwds):
        time_begin = time.time()
        ret = func(*args, **kwds)
        time_end = time.time()
        time_cost = time_end - time_begin
        print 'call %s cost %f second(s)' % (func.__name__, time_cost)
        return ret
    return wrapper
