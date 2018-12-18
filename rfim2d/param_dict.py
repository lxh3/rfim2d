import numpy as np

fit_types = ['A', 'dMdh', 'joint', 'Sigma', 'eta']


key_dict = { 
    'A':['Sigma', 'a', 'b'], 
    'dMdh':['hMax', 'eta', 'a', 'b', 'c', 'd'], 
    'joint': ['rScale', 'sScale', 'etaScale', 'sigmaNu', 'betaDeltaOverNu', 'B', 'C', 'F'], 
    'Sigma':['rScale', 'sScale', 'sigmaNu', 'B', 'F'],
    'eta':['rScale', 'etaScale', 'betaDeltaOverNu', 'B', 'C']
}


powerlaw_key_dict = {
    'joint':['rScale', 'sScale', 'etaScale', 'sigma', 'betaDelta', 'rc'], 
    'Sigma':['rScale', 'sScale', 'sigma', 'rc'], 
    'eta':['rScale', 'etaScale', 'betaDelta', 'rc']
}


def split_dict(adict):
    """
    Split a dictionary type into a list of 
    keys and values
    """
    if isinstance(adict, dict):
        keys = list(adict.keys())
        values = list(adict.values())
        return keys, values
    else:
        print('Error: Attempted to split a non-dict object into keys and values')
        return -1


def join_dict(keys, values):
    """
    Create a dictionary from a list of 
    keys and values having equal lengths
    """
    if len(keys)==len(values):
        adict = dict(zip(keys,values))
        return adict
    else:
        print('Error: Attempting to create a dictionary from a key and value list of unequal length')
        return -1


def get_keys(fit_type, func_type='wellbehaved'):
    """
    Get the list of keys corresponding to the fit parameters
    """
    if fit_type in fit_types:
        if func_type=='powerlaw':
            try:
                return powerlaw_key_dict[fit_type]
            except KeyError:
                print('Error: '+fit_type+' does not have a powerlaw form. Cannot obtain parameter keys.')
                return -1
        else:
            return key_dict[fit_type]
    else:
        print('Error: Fit type not recognized. Unable to obtain keys for parameters')
        return -1


def ensure_dict(params, key_dict, sigmaNu_fixed):
    """ 
    Convert params to dictionary format for easier access if necessary
    """
    if not isinstance(params, dict):
        if sigmaNu_fixed:
            params = list(params)
            params.insert(3,0.5)
        keys = key_dict['joint']
        params = join_dict(keys,params)
    return params


def separate_params(params, key_dict):
    """
    Separate joint params dictionary into params associated
    with Sigma and eta separately
    """
    Sigma_keys = key_dict['Sigma']
    params_Sigma = dict((k, params[k]) for k in Sigma_keys)
    eta_keys = key_dict['eta']
    params_eta = dict((k, params[k]) for k in eta_keys)
    return params_Sigma, params_eta


def format_for_residual(params_dict_Sigma, params_dict_eta, func_type):
    """
    Converts dictionaries of the parameters associated with Sigma and eta
    to lists for use by rfim2d.residuals.joint_residual()
    and calculates a prior for the eta fit
    """
    if func_type=='powerlaw':
        prior = None
    else:
        pe = params_dict_eta
        priorWeight = 0.1
        prior = priorWeight*np.array([np.log(pe['rScale']),np.log(pe['etaScale']),np.log(pe['betaDeltaOverNu']),pe['B'],pe['C']])
    params_Sigma = list(params_dict_Sigma.values())
    params_eta = list(params_dict_eta.values())
    return params_Sigma, params_eta, prior


def divvy_params(params, sigmaNu_fixed=True, func_type='wellbehaved'):
    """
    divide parameters into those associated with Sigma(r) and those associated with eta(r) 
    Input:
        params - list of all parameters found from joint fit
        sigmaNu_fixed - whether sigmaNu was set to 0.5 in fit
        func_type - form of dw/dl used - supplied to detemine number of params for each function
    Output:
        params_Sigma - params associated with Sigma(r)
        params_eta - params associated with eta(r)
        prior - prior used to keep eta fit sensible
    """

    if func_type=='powerlaw':
        local_key_dict = powerlaw_key_dict
        sigmaNu_fixed = False
    else:
        local_key_dict = key_dict

    params_dict = ensure_dict(params, local_key_dict, sigmaNu_fixed)

    params_Sigma, params_eta = separate_params(params_dict, local_key_dict)

    if not isinstance(params, dict):
        params_Sigma, params_eta, prior = format_for_residual(params_Sigma, params_eta, func_type)
    else:
        prior = None

    return params_Sigma, params_eta, prior

