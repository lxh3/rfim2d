import numpy as np
from scipy.optimize import leastsq

from .scaling import As_Scaling, dMdh_Scaling, Sigma_func, eta_func
from .print_fit_info import print_A_fit_info, print_dMdh_fit_info, print_joint_fit_info


# General fit function

def perform_fit(residual_func, params0, args):
    """
    Function to print beginning cost, perform fit and subsequently print ending cost
    Input:
        residual_func - function which provides the residual
        params0 - initial guess for parameters
        args - arguments which remain fixed in the residual function
    Output:
        params - best fit value of the parameters
        err - error in the fit
    """
    res = residual_func(params0, args)
    print('initial cost=', (res*res).sum())
    params, err = leastsq(residual_func, params0, args=args)
    res = residual_func(params, args)
    print('cost=', (res*res).sum())
    return params, err



# Functions to fit assumed scaling form for s*A(s,r) given by As_Scaling

def residual_As_Scaling(params, args):
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

def fit_As_Scaling(args, params0=None):
    """
    Perform the fit of A(s)
    Input:
        args - s_list, As_list
            s_list - list of avalanche size values obtained from simulation
            As_list - corresponding values of area weighted size distribution times avalanche size at the given values of s in s_list
        params0 - optional initial values for the constants in the functional form of As
    Output:
        params - best fit values for the constants in the functional form of dMdh
        err - error of the fit
    """
    s_list, As_list = args
    num_curves = len(s_list)
    if params0 == None:
        Sigma0 = [(As[:-1]*(s[1:]-s[:-1])).sum() for As, s in zip(As_list,s_list)]
        constants = np.array([0.8, 1.0])
        params0 = np.concatenate([Sigma0,constants])
    params, err = perform_fit(residual_As_Scaling, params0, args)
    print_A_fit_info(num_curves, params)
    return params, err



# Functions to fit assumed scaling form for eta*dMdh(h,r) given by dMdh_Scaling

def residual_dMdh_Scaling(params, args):
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

def fit_dMdh_Scaling(args, params0=None, weight_list=None):
    """
    Perform the fit of dMdh(h)
    Input:
        args - h_list, dMdh_list
            h_list - list of field values obtained from simulation
            dMdh_list - corresponding values of dM/dh at the given values of h in h_list
        params0 - optional initial values for the constants in the functional form of dMdh 
        weight_list - weighting for the importance of different curves in the fit
    Output:
        params - best fit values for the constants in the functional form of dMdh
        err - error of the fit
    """
    h_list, dMdh_list = args
    num_curves = len(h_list)
    if params0==None:
        hmax0 = np.array([h[np.argmax(dMdh)] for h, dMdh in zip(h_list, dMdh_list)])
        eta0 = np.array([np.sqrt(np.pi/2)/max(dMdh) for dMdh in dMdh_list])
        constants = np.array([0.36, 0.0, 0.36, 1.0])
        params0 = np.concatenate([hmax0,eta0,constants])
    if weight_list==None:
        weight_list = 1.0/eta0
    allargs = [h_list, dMdh_list, weight_list]
    params, err = perform_fit(residual_dMdh_Scaling, params0, allargs)
    print_dMdh_fit_info(num_curves, params)
    return params, err
    


# Function to fit nonlinear scaling variable functions Sigma(r) and eta(r), jointly

def joint_residual(params, args):
    """
    Residual for joint_fit 
    Input:
        params - suggested constants of the functions Sigma and eta to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """ 
    #Extract r, Sigma(r),  eta(r), priorWeight, and sigmaNu_fixed values from args
    r_list, Sigma, eta, priorWeight, sigmaNu_fixed = args

    #Divvy up params between sigma and eta functions
    if sigmaNu_fixed:
        sigmaNu = 0.5
        rScale, sScale, etaScale, betaDeltaOverNu, B, C, F = params
    else:
        rScale, sScale, etaScale, sigmaNu, betaDeltaOverNu, B, C, F = params
    params_sigma = rScale, sScale, sigmaNu, B, F
    params_eta = rScale, etaScale, betaDeltaOverNu, B, C

    #Calculate residual for Sigma
    residualSigma = np.log(Sigma_func(r_list,params_sigma))-np.log(Sigma)

    #Calculate residual for eta
    prior = priorWeight*np.array([np.log(rScale),np.log(etaScale),np.log(betaDeltaOverNu),B,C])
    residualeta = (eta-eta_func(r_list,params_eta))/eta
    residualeta = np.concatenate([residualeta, prior])

    #Combine residuals
    residual = np.concatenate([residualSigma, residualeta])
    return residual

def joint_fit(args, params0=None, sigmaNu_fixed=True):
    """
    Perform the joint fit of Sigma(r) and eta(r)
    Input:
        args - r_list, Sigma, eta, priorWeight
            r_list - list of r values for which there are values of Sigma and eta
            Sigma - list of Sigma values at each r
            eta - list of eta values at each r
            priorWeight - Weight to keep residual of eta from blowing up - set to 0.1 for paper
        params0 - optional initial values for the constants in the functional forms of Sigma and eta 
        sigmaNu_fixed - flag for whether to fit the value of sigmaNu or fix it to 0.5
    Output:
        params - best fit values for the constants in the functional forms of Sigma and eta
        err - error of the fit
    """
    r_list, Sigma, eta, priorWeight = args
    if params0 == None:
        if sigmaNu_fixed:
            rScale, sScale, etaScale, betaDeltaOverNu, B, C, F = [1.0, 8.0, 0.3, 1.0, 1.0, 0.1, 0.1]
            params0 = np.array([rScale, sScale, etaScale, betaDeltaOverNu, B, C, F])
        else:
            rScale, sScale, etaScale, sigmaNu, betaDeltaOverNu, B, C, F = [1.0, 8.0, 0.3, 1.0, 1.0, 1.0, 0.1, 0.1]
            params0 = np.array([rScale, sScale, etaScale, sigmaNu, betaDeltaOverNu, B, C, F])
    fullargs = [r_list, Sigma, eta, priorWeight, sigmaNu_fixed]
    params, err = perform_fit(joint_residual, params0, fullargs)
    print_joint_fit_info(params, sigmaNu_fixed)
    return params, err

