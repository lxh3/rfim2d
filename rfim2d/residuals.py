import numpy as np

from .scaling import As_Scaling, dMdh_Scaling, Sigma_func, eta_func
from .param_dict import divvy_params

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
        residuals = np.append(residuals, np.log(As) - np.log(As_Scaling(s, Sigma, constants)))

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

    for h, dMdh, hmax, eta, weight in zip(h_list, dMdh_list, hmax_list, eta_list, weight_list):
        residuals = np.append(residuals, (dMdh - dMdh_Scaling(h, [hmax, eta], constants))*weight)

    return np.array(residuals)

def Sigma_residual(params, args):
    """
    Residual for Sigma_fit 
    Input:
        params - suggested constants of the function Sigma to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """
    r_list, Sigma, sigmaNu_fixed, func_type = args

    if sigmaNu_fixed and func_type!='powerlaw':
        params_Sigma = list(params)
        params_Sigma.insert(2,0.5)
    else:
        params_Sigma = params

    return np.log(Sigma_func(r_list,params_Sigma,func_type=func_type))-np.log(Sigma)

def eta_residual(params, args):
    """
    Residual for eta_fit 
    Input:
        params - suggested constants of the function eta to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """
    r_list, eta, func_type = args
    if func_type=='powerlaw':
        residual = (eta-eta_func(r_list,params,func_type=func_type))/eta
    else:
        rScale,etaScale,betaDeltaOverNu,B,C = params
        priorWeight = 0.1
        prior = priorWeight*np.array([np.log(rScale),np.log(etaScale),np.log(betaDeltaOverNu),B,C])
        residualeta = (eta-eta_func(r_list,params,func_type=func_type))/eta
        residual = np.concatenate([residualeta,prior])
    return residual

def joint_residual(params, args):
    """
    Residual for joint_fit 
    Input:
        params - suggested constants of the functions Sigma and eta to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """
    rA, Sigma, rdMdh, eta, sigmaNu_fixed, func_type = args

    params_Sigma, params_eta, prior = divvy_params(params, sigmaNu_fixed, func_type)

    residualSigma = np.log(Sigma_func(rA,params_Sigma,func_type=func_type))-np.log(Sigma)

    residualeta = (eta-eta_func(rdMdh,params_eta,func_type=func_type))/eta
    if func_type != 'powerlaw':
        residualeta = np.concatenate([residualeta, prior])

    residual = np.concatenate([residualSigma, residualeta])
    return residual

