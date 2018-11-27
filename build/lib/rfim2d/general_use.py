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

def load_svA():
    """
    Loads default (provided) simulation data of the area weighted avalanche size distribution function
    Output:
        data - [r, s, A, As]
            r - list of value of r simulated
            s - list of avalanche sizes obtained for each value of r simulated 
                (e.g. s[0] contains the sizes obtained with disorder=r[0])
            A - list of areas weighted size distribution function values associated with the
                corresponding value of s
            As - A*s provided for convenience
    """
    filename = pkg_resources.resource_filename(__name__, 'data/svA.pkl.gz')
    data = load_func(filename)
    return data
   
def load_hvdMdh():
    """
    Loads default (provided) simulation data of the derivative of the magnetization with respect to the field
    Output:
        data - [r, h, dMdh]
            r - list of value of r simulated
            h - list of field values obtained for each value of r simulated 
                (e.g. h[0] contains the sizes obtained with disorder=r[0])
            dMdh - list of derivatives of the megnetization with respect to the field associated with the
                   corresponding value of h
    """

    filename = pkg_resources.resource_filename(__name__, 'data/hvdMdh.pkl.gz')
    data = load_func(filename)
    return data
 
def generate_colors(num_colors):
    """
    Randomly generates color
    Input:
        num_colors - number of randomly generated colors requested
    Output:
        color_list - list of size [num_colors, 3] containing the RBG values for each color requested
    """
    c = np.random.rand(3,num_colors)
    color_list = c.T
    return color_list

