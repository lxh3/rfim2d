import numpy as np
from numpy import log

from .scaling import As_Scaling, dMdh_Scaling, Sigma_func, eta_func
from .param_dict import separate_params, generate_dict_with_fixed_params, split_dict_with_fixed_params


def As_residual(params, args):
    """
    Residual for fit_As_Scaling
    Input:
        params - suggested constants of the function As to be fit
        args - arguments which remain fixed (e.g. s values)
    Output:
        residual - residual of the fit for the given parameters
    """
    s_list, As_list = args
    num_curves = len(s_list)
    Sigma_list = params[:num_curves]
    constants = params[num_curves:]
    residuals = np.array([])

    for s, As, Sigma in zip(s_list, As_list, Sigma_list):
        As_residual = log(As) - log(As_Scaling(s, Sigma, constants))
        residuals = np.append(residuals, As_residual)

    return np.array(residuals)


def dMdh_residual(params, args):
    """
    Residual for fit_dMdh_Scaling
    Input:
        params - suggested constants of the function of dMdh to be fit
        args - arguments which remain fixed (e.g. h values)
    Output:
        residual - residual of the fit for the given parameters
    """
    h_list, dMdh_list, weight_list = args
    num_curves = len(h_list)
    hmax_list = params[:num_curves]
    eta_list = params[num_curves:2*num_curves]
    constants = params[2*num_curves:]
    residuals = np.array([])

    zipped = zip(h_list, dMdh_list, hmax_list, eta_list, weight_list)
    for h, dMdh, hmax, eta, weight in zipped:
        dMdh_residual = (dMdh - dMdh_Scaling(h, [hmax, eta], constants))*weight
        residuals = np.append(residuals, dMdh_residual)

    return np.array(residuals)


def Sigma_residual(params, args):
    """
    Residual for Sigma_fit
    Input:
        params - suggested parameters of the function Sigma to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """
    r_list, Sigma, keys, fixed_dict, func_type = args
    param_dict = generate_dict_with_fixed_params(params, keys, fixed_dict)
    Sigma_func_val = Sigma_func(r_list, param_dict, func_type=func_type)
    return log(Sigma_func_val)-log(Sigma)


def eta_prior(param_dict):
    """
    Compute prior for non-'power law' fits
    """
    pe = param_dict
    priorWeight = 0.1
    scaled_values = np.array([log(pe['rScale']), log(pe['etaScale']),
                              log(pe['lambdaH']), pe['B'], pe['F']])
    prior = priorWeight*scaled_values
    return prior
    

def eta_residual(params, args):
    """
    Residual for eta_fit
    Input:
        param_dict - suggested parameters of the function eta to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """
    r_list, eta, keys, fixed_dict, func_type = args
    param_dict = generate_dict_with_fixed_params(params, keys, fixed_dict)

    residualeta = (eta-eta_func(r_list, param_dict, func_type=func_type))/eta

    if func_type != 'power law':
        prior = eta_prior(param_dict)
        residualeta = np.concatenate([residualeta, prior])

    return residualeta


def joint_residual(params, args):
    """
    Residual for joint_fit
    Input:
        param_dict - suggested parameters of the functions Sigma and eta to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """
    rA, Sigma, rdMdh, eta, keys, fixed_dict, func_type = args

    param_dict = generate_dict_with_fixed_params(params, keys, fixed_dict)

    params_Sigma, params_eta = separate_params(param_dict, func_type=func_type)

    Sigma_func_val = Sigma_func(rA, params_Sigma, func_type=func_type)
    residualSigma = log(Sigma_func_val)-log(Sigma)

    residualeta = (eta-eta_func(rdMdh, params_eta, func_type=func_type))/eta
    if func_type != 'power law':
        prior = eta_prior(param_dict)
        residualeta = np.concatenate([residualeta, prior])

    residual = np.concatenate([residualSigma, residualeta])
    return residual


def linear_function(x, params):
    A, B = params
    return A*x + B


def linear_residual(params, args):
    r, pS = args
    r = np.asarray(r)
    residual = np.asarray(1./log(Sigma_func(r,pS))) - np.asarray(linear_function(r, params))
    return residual

