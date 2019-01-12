import numpy as np
from scipy.optimize import leastsq

from .save_and_load import save_func, load_svA, load_hvdMdh
from .param_dict import split_dict, join_dict, get_keys, divvy_params
from .residuals import As_residual, dMdh_residual
from .residuals import Sigma_residual, eta_residual, joint_residual
from .print_fit_info import print_fit_info

params0_dict = {
    'Sigma_powerlaw': [1.0, 1.0, 0.1, 0.0],
    'Sigma_simple': [1.0, 10.0, 0.5, 0.1, 1.0],
    'Sigma_simple_sigmaNu_fixed': [1.0, 10.0, 0.1, 1.0],
    'Sigma_wellbehaved': [1.0, 10.0, 0.5, 0.1, 1.0],
    'Sigma_wellbehaved_sigmaNu_fixed': [1.0, 10.0, 0.1, 1.0],
    'Sigma_pitchfork': [1.0, 1.0, 0.5, 10.0, 1.0],
    'eta_powerlaw': [1.0, 0.3, 1.0, 0.0],
    'eta_simple': [1.0, 0.3, 1.0, 1.0, 0.1],
    'eta_wellbehaved': [1.0, 0.3, 1.0, 1.0, 0.1],
    'eta_pitchfork': [5.0, 130.0, 0.6, 4.0, 1.8],
    'joint_powerlaw': [1.0, 1.0, 1.0, 0.1, 1.0, 0.0],
    'joint_simple': [1.0, 10., 1.0, 1.0, 1.0, 0.1, 1.0, 1.0],
    'joint_simple_sigmaNu_fixed': [1.0, 10., 1.0, 1.0, 0.1, 1.0, 1.0],
    'joint_wellbehaved': [1.0, 10., 1.0, 1.0, 1.0, 0.1, 1.0, 1.0],
    'joint_wellbehaved_sigmaNu_fixed': [1.0, 10., 1.0, 1.0, 0.1, 1.0, 1.0],
    'joint_pitchfork': [5.0, -1.0, 100.0, 0.5, 1.0, 0.0, 10.0, 2.0]
}


def perform_fit(residual_func, params0, args, verbose=False):
    """
    Function to print beginning cost, perform fit and subsequently
    print ending cost
    Input:
        residual_func - function which provides the residual
        params0 - initial guess for parameters
        args - arguments which remain fixed in the residual function
        verbose - flag to print cost
    Output:
        params - best fit value of the parameters
        err - error in the fit
    """
    if isinstance(params0, dict):
        keys, params0 = split_dict(params0)

    res = residual_func(params0, args)
    if verbose:
        print('initial cost=', (res*res).sum())
    params, err = leastsq(residual_func, params0, args=args)
    res = residual_func(params, args)
    if verbose:
        print('cost=', (res*res).sum())

    try:
        params = join_dict(keys, params)
    except NameError:
        pass

    return params, err


def save_and_show(params, fit_type, func_type='wellbehaved',
                  filename=None, show=True):
    """
    Save fit parameters and/or plot fit information in figure format
    """
    if filename is not None:
        save_func(filename+'.pkl.gz', params)
        print_fit_info(params, fit_type, func_type=func_type,
                       filename=filename+'.png', show=False)

    if show:
        print_fit_info(params, fit_type, func_type=func_type, show=True)

    return


def fit_As_Scaling(args, params0=None, filename=None,
                   verbose=False, show_params=True):
    """
    Perform the fit of A(s)
    Input:
        args - s_list, As_list
            s_list - list of avalanche size values obtained from simulation
            As_list - corresponding values of area weighted size
                      distribution times avalanche size at the
                      given values of s in s_list
        params0 - optional initial values for the constants in the
                  functional form of As
        filename - data is saved under 'filename' if a file name is given
        verbose - flag to print cost
        show_params - flag for whether to plot parameter names
                      with their values
    Output:
        params - best fit values for the constants in the functional
                 form of dMdh
        err - error of the fit
    """
    s_list, As_list = args
    num_curves = len(s_list)

    if params0 is None:
        zipped = zip(As_list, s_list)
        Sigma0 = [(As[:-1]*(s[1:]-s[:-1])).sum() for As, s in zipped]
        constants = np.array([0.8, 1.0])
        params0 = np.concatenate([Sigma0, constants])

    params, err = perform_fit(As_residual, params0, args, verbose=verbose)

    Sigma = params[:num_curves]
    a, b = params[num_curves:]

    keys = get_keys('A')
    values = [Sigma, a, b]
    params = dict(zip(keys, values))

    save_and_show(params, 'A', filename=filename, show=show_params)

    return params, err


def fit_dMdh_Scaling(args, params0=None, weight_list=None, filename=None,
                     verbose=False, show_params=True):
    """
    Perform the fit of dMdh(h)
    Input:
        args - h_list, dMdh_list
            h_list - list of field values obtained from simulation
            dMdh_list - corresponding values of dM/dh at the given values
                        of h in h_list
        params0 - optional initial values for the constants in the
                  functional form of dMdh
        weight_list - weighting for the importance of different
                      curves in the fit
        filename - data is saved under 'filename' if a file name is given
        verbose - flag to print cost
        show_params - flag for whether to plot parameter names
                      with their values
    Output:
        params - best fit values for the constants in the functional
                 form of dMdh
        err - error of the fit
    """
    h_list, dMdh_list = args
    num_curves = len(h_list)

    if params0 is None:
        zipped = zip(h_list, dMdh_list)
        hmax0 = np.array([h[np.argmax(dMdh)] for h, dMdh in zipped])
        eta0 = np.array([np.sqrt(np.pi/2)/max(dMdh) for dMdh in dMdh_list])
        constants = np.array([0.36, 0.0, 0.36, 1.0])
        params0 = np.concatenate([hmax0, eta0, constants])

    if weight_list is None:
        weight_list = 1.0/eta0

    allargs = [h_list, dMdh_list, weight_list]
    params, err = perform_fit(dMdh_residual, params0, allargs,
                              verbose=verbose)

    hMax = params[:num_curves]
    eta = params[num_curves:2*num_curves]
    a, b, c, d = params[2*num_curves:]

    keys = get_keys('dMdh')
    values = [hMax, eta, a, b, c, d]
    params = dict(zip(keys, values))

    save_and_show(params, 'dMdh', filename=filename, show=show_params)

    return params, err


def get_Sigma(filename=None, data=None, show_params=False):
    """
    Perform the fit of A and return (r, Sigma) pairs
    """
    if data is None:
        r, s, A, As = load_svA(filename)
    else:
        r, s, A, As = data
    params, err = fit_As_Scaling([s, As], show_params=show_params)
    return r, params['Sigma']


def Sigma_fit(args, params0=None, sigmaNu_fixed=True,
              func_type='wellbehaved', filename=None,
              verbose=False, show_params=True):
    """
    Perform the fit of Sigma(r)
    Input:
        args - r_list, Sigma
            r_list - list of r values for which there are
                     values of Sigma and eta
            Sigma - list of Sigma values at each r
        params0 - optional initial values for the constants
                  in the functional form of Sigma
        sigmaNu_fixed - flag for whether to fit the value of
                        sigmaNu or fix it to 0.5
        func_type - which form of Sigma(r) to use. Options:
            'powerlaw' - Sigma(r) derived with dw/dl = (1/nu) w
            'simple' - Sigma(r) derived with dw/dl = w^2 + B w^3
            'wellbehaved' - Sigma(r) derived with dw/dl = w^2 / (1 + B w)
            'pitchfork' - Sigma(r) derived with dw/dl = w^3 + B w^5
        filename - data is saved under 'filename' if a file name is given
        verbose - flag to print cost
        show_params - flag for whether to plot parameter names with
                      their values
    Output:
        params - best fit values for the constants in the functional
                 form of Sigma
        err - error of the fit
    """
    r_list, Sigma = args

    if func_type == 'powerlaw' or func_type == 'pitchfork':
        sigmaNu_fixed = False

    if params0 is None:
        if sigmaNu_fixed:
            params0_key = 'Sigma_'+func_type+'_sigmaNu_fixed'
        else:
            params0_key = 'Sigma_'+func_type
        params0 = params0_dict[params0_key]

    fullargs = [r_list, Sigma, sigmaNu_fixed, func_type]
    params, err = perform_fit(Sigma_residual, params0, fullargs,
                              verbose=verbose)

    keys = get_keys('Sigma', func_type=func_type)
    if sigmaNu_fixed and func_type != 'powerlaw':
        params = list(params)
        params.insert(2, 0.5)
    params = dict(zip(keys, params))

    save_and_show(params, 'Sigma', func_type=func_type,
                  filename=filename, show=show_params)

    return params, err


def get_eta(filename=None, data=None, show_params=False):
    """
    Perform the fit of dMdh and return (r, eta) pairs
    """
    if data is None:
        r, h, dMdh = load_hvdMdh(filename)
    else:
        r, h, dMdh = data
    params, err = fit_dMdh_Scaling([h, dMdh], show_params=show_params)
    return r, params['eta']


def eta_fit(args, params0=None, func_type='wellbehaved', filename=None,
            verbose=False, show_params=True):
    """
    Perform the fit of eta(r)
    Input:
        args - r_list, eta
            r_list - list of r values for which there are values
                     of Sigma and eta
            eta - list of eta values at each r
        params0 - optional initial values for the constants in the
                  functional form of eta
        func_type - which form of eta(r) to use. Options:
            'powerlaw' - eta(r) derived with dw/dl = (1/nu) w
            'simple' - eta(r) derived with dw/dl = w^2 + B w^3
            'wellbehaved' - eta(r) derived with dw/dl = w^2 / (1 + B w)
            'pitchfork' - eta(r) derived with dw/dl = w^3 + B w^5
        filename - data is saved under 'filename' if a file name is given
        verbose - flag to print cost
        show_params - flag for whether to plot parameter names
                      with their values
    Output:
        params - best fit values for the constants in the functional
                 form of eta
        err - error of the fit
    """
    r_list, eta = args

    if params0 is None:
        params0_key = 'eta_'+func_type
        params0 = params0_dict[params0_key]

    fullargs = [r_list, eta, func_type]
    params, err = perform_fit(eta_residual, params0, fullargs, verbose=verbose)

    keys = get_keys('eta', func_type=func_type)
    values = params
    params = dict(zip(keys, values))

    save_and_show(params, 'eta', func_type=func_type,
                  filename=filename, show=show_params)

    return params, err


def joint_fit(args, params0=None, sigmaNu_fixed=True,
              func_type='wellbehaved', filename=None,
              verbose=False, show_params=True):
    """
    Perform the joint fit of Sigma(r) and eta(r)
    Input:
        args - rA, Sigma, rdMdh, eta
            rA - list of r values for which there are values Sigma
            Sigma - list of Sigma values at each r
            rdMdh - list of r values for which there are values of eta
            eta - list of eta values at each r
        params0 - optional initial values for the constants in the
                  functional forms of Sigma and eta
        sigmaNu_fixed - flag for whether to fit the value of sigmaNu
                        or fix it to 0.5
        func_type - which form of eta(r) to use. Options:
            'powerlaw' - eta(r) derived with dw/dl = (1/nu) w
            'simple' - eta(r) derived with dw/dl = w^2 + B w^3
            'wellbehaved' - eta(r) derived with dw/dl = w^2 / (1 + B w)
            'pitchfork' - eta(r) derived with dw/dl = w^3 + B w^5
        filename - data is saved under 'filename' if a file name is given
        verbose - flag to print cost
        show_params - flag for whether to plot parameter names
                      with their values
    Output:
        params - best fit values for the constants in the functional
                 forms of Sigma and eta
        err - error of the fit
    """
    rA, Sigma, rdMdh, eta = args

    if func_type == 'powerlaw' or func_type == 'pitchfork':
        sigmaNu_fixed = False

    if params0 is None:
        if sigmaNu_fixed:
            params0_key = 'joint_'+func_type+'_sigmaNu_fixed'
        else:
            params0_key = 'joint_'+func_type
        params0 = params0_dict[params0_key]

    fullargs = [rA, Sigma, rdMdh, eta, sigmaNu_fixed, func_type]
    params, err = perform_fit(joint_residual, params0, fullargs,
                              verbose=verbose)

    keys = get_keys('joint', func_type=func_type)
    if sigmaNu_fixed and func_type != 'powerlaw':
        params = list(params)
        params.insert(3, 0.5)
    params = dict(zip(keys, params))

    save_and_show(params, 'joint', func_type=func_type,
                  filename=filename, show=show_params)

    return params, err


# Perform all fits (A, dMdh, and Sigma and eta jointly)
#  and return parameters

def perform_all_fits(filenames=[None, None], data=None, sigmaNu_fixed=True,
                     func_type='wellbehaved', verbose=False,
                     show_params=True):
    """
    Perform the fit of A, dMdh, Sigma and eta and return values
    Input:
        filenames - [A_filename, dMdh_filename]
                  - filenames to load data from
            A_filename - location where area weighted
                         size distribution is stored
            dMdh_filename - location where dM/dh data is stored
        data - [dataA, datadMdh] - if filenames==None, checks to see if
                                   data was provided here manually
            dataA - [r, s, A, As]
            datadMdh - [r, h, dMdh]
        sigmaNu_fixed - flag for whether to fit the value of sigmaNu
                        or fix it to 0.5
        func_type - which form of Sigma(r) to use. Options:
            'powerlaw' - Sigma(r) derived with dw/dl = (1/nu) w
            'simple' - Sigma(r) derived with dw/dl = w^2 + B w^3
            'wellbehaved' - Sigma(r) derived with dw/dl = w^2 / (1 + B w)
            'pitchfork' - Sigma(r) derived with dw/dl = w^3 + B w^5
        save - whether to save the fit data
        verbose - flag to print cost
        show_params - flag for whether to plot parameter names
                      with their values
    Output:
        params_A - fit values found for A
        params_dMdh - fit values found for dM/dh
        params_Sigma - fit values found for Sigma
        params_eta - fit values found for eta
    """
    if func_type == 'powerlaw' or func_type == 'pitchfork':
        sigmaNu_fixed = False

    if data is None:
        rA, s, A, As = load_svA(filename=filenames[0])
        rdMdh, h, dMdh = load_hvdMdh(filename=filenames[1])
    else:
        rA, s, A, As = data[0]
        rdMdh, h, dMdh = data[1]

    args_A = [s, As]
    params_A, err = fit_As_Scaling(args_A, verbose=verbose,
                                   show_params=show_params)
    Sigma = params_A['Sigma']

    args_dMdh = [h, dMdh]
    params_dMdh, err = fit_dMdh_Scaling(args_dMdh, verbose=verbose,
                                        show_params=show_params)
    eta = params_dMdh['eta']

    args = [rA, Sigma, rdMdh, eta]
    params, err = joint_fit(args, sigmaNu_fixed=sigmaNu_fixed,
                            func_type=func_type, verbose=verbose,
                            show_params=show_params)
    divvyed_params = divvy_params(params, sigmaNu_fixed=sigmaNu_fixed,
                                  func_type=func_type)
    params_Sigma, params_eta, prior = divvyed_params

    return params_A, params_dMdh, params_Sigma, params_eta
