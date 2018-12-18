import gzip
import pickle
import numpy as np
import pkg_resources 

def save_func(fn, one_thing):
    """
    Save one thing with pickle and gzip
    Input:
        fn - filename to save to
        one_thing - thing to save
    """
    with gzip.open(fn,'wb') as f:
        pickle.dump(one_thing,f,-1)

def load_func(fn):
    """
    Load one thing saved with pickle and gzip
    Input:
        fn - filename to load from
    Output:
        one_thing - thing loaded from file
    """
    with gzip.open(fn) as f:
        one_thing = pickle.load(f,encoding='latin1')
    return one_thing

def load_svA(filename=None):
    """
    Loads simulation data of the area weighted avalanche size distribution function
    from default (provided) file unless filename for source provided
    Input:
        filename - optional location to load data from
    Output:
        data - [r, s, A, As]
            r - list of value of r simulated
            s - list of avalanche sizes obtained for each value of r simulated 
                (e.g. s[0] contains a list of the sizes obtained with disorder=r[0])
            A - list of areas weighted size distribution function values associated with the
                corresponding value of s
            As - A*s provided for convenience
    """
    if filename == None:
        filename = pkg_resources.resource_filename(__name__, 'data/svA.pkl.gz')
    data = load_func(filename)
    return data
   
def load_hvdMdh(filename=None):
    """
    Loads simulation data of the derivative of the magnetization with respect to the field
    from default (provided) file unless filename for source provided
    Input:
        filename - optional location to load data from
    Output:
        data - [r, h, dMdh]
            r - list of value of r simulated
            h - list of field values obtained for each value of r simulated 
                (e.g. h[0] contains a list of the sizes obtained with disorder=r[0])
            dMdh - list of derivatives of the megnetization with respect to the field associated with the
                   corresponding value of h
    """
    if filename == None:
        filename = pkg_resources.resource_filename(__name__, 'data/hvdMdh.pkl.gz')
    data = load_func(filename)
    return data

