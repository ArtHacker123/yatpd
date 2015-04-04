# -*- coding: utf-8 -*-

import time


def timer(func, *args, **kwds):
    '''A simple decorator to count time.
    '''
    time_begin = time.time()
    ret = func(*args, **kwds)
    time_end = time.time()
    time_cost = time_begin - time_end
    print 'call %s cost %f second(s)' % (func.__name__, time_cost)
    return ret
