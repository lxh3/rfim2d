import numpy as np

fit_types = ['A', 'dMdh', 'joint', 'Sigma', 'eta']


key_dict = {
    'A': ['Sigma', 'a', 'b'],
    'dMdh': ['hMax', 'eta', 'a', 'b', 'c', 'd'],
    'joint': [
        'rScale', 'rc', 'sScale', 'etaScale', 'df',
        'lambdaH', 'B', 'C', 'F'
        ],
    'Sigma': ['rScale', 'rc', 'sScale', 'df', 'B', 'C'],
    'eta': ['rScale', 'rc', 'etaScale', 'lambdaH', 'B', 'F']
}


powerlaw_key_dict = {
    'joint': ['rScale', 'rc', 'sScale', 'etaScale', 'sigma', 'betaDelta'],
    'Sigma': ['rScale', 'rc', 'sScale', 'sigma'],
    'eta': ['rScale', 'rc', 'etaScale', 'betaDelta']
}


def split_dict(adict):
    """
    Split a dictionary type into a list of keys and values
    """
    if isinstance(adict, dict):
        keys = list(adict.keys())
        values = list(adict.values())
        return keys, values
    else:
        print('Error: Attempted to split a non-dict '
              'object into keys and values')
        return -1


def join_dict(keys, values):
    """
    Create a dictionary from a list of
    keys and values having equal lengths
    """
    if len(keys) == len(values):
        adict = dict(zip(keys, values))
        return adict
    else:
        print('Error: Attempting to create a dictionary from '
              'a key and value list of unequal length')
        return -1


def get_keys(fit_type, func_type='well-behaved'):
    """
    Get the list of keys corresponding to the fit parameters
    """
    if fit_type in fit_types:

        if func_type == 'power law':
            try:
                keys = powerlaw_key_dict[fit_type]
            except KeyError:
                print('Error: '+fit_type+' does not have a power law form. '
                      ' Cannot obtain parameter keys.')
                return -1

        else:
            keys = key_dict[fit_type]

        return keys

    else:
        print('Error: Fit type not recognized. '
              'Unable to obtain keys for parameters')
        return -1


def separate_params(param_dict, func_type='well-behaved'):
    """
    Separate joint params dictionary into params associated
    with Sigma and eta separately
    """
    Sigma_keys = get_keys('Sigma', func_type=func_type)
    eta_keys = get_keys('eta', func_type=func_type)

    params_Sigma = dict((k, param_dict[k]) for k in Sigma_keys)
    params_eta = dict((k, param_dict[k]) for k in eta_keys)

    return params_Sigma, params_eta


def generate_dict_with_fixed_params(params, keys, fixed_dict):
    if fixed_dict is not None:
        new_param_list = []
        i = 0
        for key in keys:
            try:
                p = fixed_dict[key]
            except KeyError:
                p = params[i]
                i += 1
            new_param_list.append(p)
    else:
        new_param_list = params
    param_dict = join_dict(keys, new_param_list)
    return param_dict


def split_dict_with_fixed_params(params_dict, fixed_dict):
    temp_dict = params_dict.copy()
    keys, values = split_dict(fixed_dict)
    for key in keys:
        try:
            temp_dict.pop(key)
        except KeyError:
            print(key+' does not exist, cannot remove value')
    keys, values = split_dict(temp_dict)
    return values
