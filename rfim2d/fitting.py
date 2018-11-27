import numpy as np
from scipy.optimize import leastsq

from .general_use import load_func, load_svA, load_hvdMdh
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
   



# Function to fit nonlinear scaling variable function Sigma(r) by itself

def add_sigmaNu(params):
    """
    Add sigmaNu to list of parameters obtained with
    a fit where sigmaNu is fixed to 0.5 (sigmaNu_fixed=True)
    Input:
        params - params from fit with fixed sigmaNu
    Output:
        params_Sigma - full list of params with sigmaNu included
    """
    rScale, sScale, B, F = params
    sigmaNu = 0.5
    params_Sigma = rScale, sScale, sigmaNu, B, F
    return params_Sigma

def Sigma_residual(params, args):
    """
    Residual for Sigma_fit 
    Input:
        params - suggested constants of the function Sigma to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """
    r_list, Sigma, sigmaNu_fixed, simple = args
    if sigmaNu_fixed:
        sigmaNu = 0.5
        rScale, sScale, B, F = params
    else:
        rScale, sScale, sigmaNu, B, F = params
    params_Sigma = rScale, sScale, sigmaNu, B, F
    return np.log(Sigma_func(r_list,params_Sigma,simple=simple))-np.log(Sigma)

def Sigma_fit(args, params0=None, sigmaNu_fixed=True, simple=False):
    """
    Perform the fit of Sigma(r) 
    Input:
        args - r_list, Sigma
            r_list - list of r values for which there are values of Sigma and eta
            Sigma - list of Sigma values at each r
        params0 - optional initial values for the constants in the functional form of Sigma  
        sigmaNu_fixed - flag for whether to fit the value of sigmaNu or fix it to 0.5
        simple - flag for whether to use the Sigma derived from the simplest dw/dl
    Output:
        params - best fit values for the constants in the functional form of Sigma
        err - error of the fit
    """
    r_list, Sigma = args

    if params0 == None:

        rScale, sScale, B, F = [1.0, 1.0, -1.0 , 1.0]
 
        if sigmaNu_fixed:
            params0 = np.array([rScale, sScale, B, F])
        else:
            sigmaNu = 0.5
            params0 = np.array([rScale, sScale, sigmaNu, B, F])

    fullargs = [r_list, Sigma, sigmaNu_fixed, simple]
    params, err = perform_fit(Sigma_residual, params0, fullargs)
    #print_Sigma_fit_info(params, sigmaNu_fixed)#Add this later
    if sigmaNu_fixed:
        params = add_sigmaNu(params)
    return params, err



# Function to fit nonlinear scaling varibale function eta(r) by itself

def eta_residual(params, args):
    """
    Residual for eta_fit 
    Input:
        params - suggested constants of the function eta to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """
    rScale,etaScale,betaDeltaOverNu,B,C = params
    r_list, eta, priorWeight = args
    prior = priorWeight*np.array([np.log(rScale),np.log(etaScale),np.log(betaDeltaOverNu),B,C])
    residualeta = (eta-eta_func(r_list,params))/eta
    return np.concatenate([residualeta,prior])

def eta_fit(args, params0=None):
    """
    Perform the fit of eta(r) 
    Input:
        args - r_list, eta, priorWeight
            r_list - list of r values for which there are values of Sigma and eta
            eta - list of eta values at each r
            priorWeight - Weight to keep residual of eta from blowing up - set to 0.1 for paper
        params0 - optional initial values for the constants in the functional form of eta 
    Output:
        params - best fit values for the constants in the functional form of eta
        err - error of the fit
    """
    r_list, eta, priorWeight = args

    if params0 == None:
        params0 = [1.0, 0.3, 1.0, 1.0, 0.1]

    params, err = perform_fit(eta_residual, params0, args)
    #print_eta_fit_info(params)#Add this later
    return params, err




# Function to fit nonlinear scaling variable functions Sigma(r) and eta(r), jointly

def divvy_params(params, sigmaNu_fixed, priorWeight=0.1):
    """
    divide parameters into those associated with Sigma(r) and those associated with eta(r) 
    Input:
        params - list of all parameters found from joint fit
        sigmaNu_fixed - whether sigmaNu was set to 0.5 in fit
        priorWeight - weight on prior used to keep eta fit sensible
    Output:
        params_Sigma - params associated with Sigma(r)
        params_eta - params associated with eta(r)
        prior - prior used to keep eta fit sensible
    """
    if sigmaNu_fixed:
        sigmaNu = 0.5
        rScale, sScale, etaScale, betaDeltaOverNu, B, C, F = params
    else:
        rScale, sScale, etaScale, sigmaNu, betaDeltaOverNu, B, C, F = params
    params_Sigma = rScale, sScale, sigmaNu, B, F
    params_eta = rScale, etaScale, betaDeltaOverNu, B, C
    prior = priorWeight*np.array([np.log(rScale),np.log(etaScale),np.log(betaDeltaOverNu),B,C])
    return params_Sigma, params_eta, prior

def joint_residual(params, args):
    """
    Residual for joint_fit 
    Input:
        params - suggested constants of the functions Sigma and eta to be fit
        args - arguments which remain fixed (e.g. r values)
    Output:
        residual - residual of the fit for the given parameters
    """ 
    #Extract r, Sigma(r),  eta(r), priorWeight, sigmaNu_fixed, and simple values from args
    r_list, Sigma, eta, priorWeight, sigmaNu_fixed, simple = args

    #Divvy up params between sigma and eta functions
    params_Sigma, params_eta, prior = divvy_params(params, sigmaNu_fixed)

    #Calculate residual for Sigma
    residualSigma = np.log(Sigma_func(r_list,params_Sigma,simple=simple))-np.log(Sigma)

    #Calculate residual for eta
    residualeta = (eta-eta_func(r_list,params_eta))/eta
    residualeta = np.concatenate([residualeta, prior])

    #Combine residuals
    residual = np.concatenate([residualSigma, residualeta])
    return residual

def joint_fit(args, params0=None, sigmaNu_fixed=True, simple=False):
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
        simple - flag for whether to use the Sigma derived from the simplest dw/dl
    Output:
        params - best fit values for the constants in the functional forms of Sigma and eta
        err - error of the fit
    """
    r_list, Sigma, eta, priorWeight = args
    if params0 == None:

        if simple:
            sScale, B, F = [1.0, -1.0, 1.0]
        else:
            sScale, B, F = [8.0, -1.0, 0.1] 

        if sigmaNu_fixed:
            rScale, etaScale, betaDeltaOverNu, C = [1.0, 0.3, 1.0, 0.1]
            params0 = np.array([rScale, sScale, etaScale, betaDeltaOverNu, B, C, F])
        else:
            rScale, etaScale, sigmaNu, betaDeltaOverNu, C = [1.0, 0.3, 1.0, 1.0, 0.1]
            params0 = np.array([rScale, sScale, etaScale, sigmaNu, betaDeltaOverNu, B, C, F])

    fullargs = [r_list, Sigma, eta, priorWeight, sigmaNu_fixed, simple]
    params, err = perform_fit(joint_residual, params0, fullargs)
    print_joint_fit_info(params, sigmaNu_fixed)
    return params, err


# Function to perform all of the fits for given data files

def perform_all_fits(A_filename=None, dMdh_filename=None, sigmaNu_fixed=True, simple=False, priorWeight=0.1):
    """
    Perform the fit of A, dMdh, Sigma and eta and return values
    Input:
        A_filename - location where area weighted size distribution is stored
        dMdh_filename - location where dM/dh data is stored
        sigmaNu_fixed - flag for whether to fit the value of sigmaNu or fix it to 0.5
        simple - flag for whether to use the Sigma derived from the simplest dw/dl
    Output:
        params_A - fit values found for A 
        params_dMdh - fit values found for dM/dh
        params_Sigma - fit values found for Sigma
        params_eta - fit values found for eta
    """

    if A_filename == None:
        r,s,A,As = load_svA()
    else:
        r,s,A,As = load_func(A_filename)

    if dMdh_filename == None:
        r,h,dMdh = load_hvdMdh()
    else:
        r,h,dMdh = load_func(dMdh_filename)

    args_A = [s, As]
    params_A, err = fit_As_Scaling(args_A)
    Sigma = params_A[:len(r)]
    
    args_dMdh = [h,dMdh]
    params_dMdh, err = fit_dMdh_Scaling(args_dMdh)
    eta = params_dMdh[len(r):2*len(r)]

    args = [r, Sigma, eta, priorWeight]
    params, err = joint_fit(args, sigmaNu_fixed=sigmaNu_fixed, simple=simple)
    params_Sigma, params_eta, prior = divvy_params(params, sigmaNu_fixed, priorWeight=priorWeight)

    return params_A, params_dMdh, params_Sigma, params_eta

# Get Sigma
 
def get_Sigma(A_filename=None):
    """
    Perform the fit of A and return (r, Sigma) pairs
    Input:
        A_filename - location where area weighted size distribution is stored
    Output:
        r - disorders simulated
        Sigma - Sigma values as a function of r determined from fit of As_Scaling
    """

    if A_filename == None:
        r,s,A,As = load_svA()
    else:
        r,s,A,As = load_func(A_filename)

    args_A = [s, As]
    params_A, err = fit_As_Scaling(args_A)
    Sigma = params_A[:len(r)]
    return r, Sigma         

# Get eta

def get_eta(dMdh_filename=None):
    """
    Perform the fit of dMdh and return (r, eta) pairs
    Input:
        dMdh_filename - location where dM/dh simulation data is stored
    Output:
        r - disorders simulated
        Sigma - eta values as a function of r determined from fit of dMdh_Scaling
    """

    if dMdh_filename == None:
        r,h,dMdh = load_hvsdMdh()
    else:
        r,h,dMdh = load_func(dMdh_filename)

    args_dMdh = [h, dMdh]
    params_dMdh, err = fit_dMdh_Scaling(args_dMdh)
    eta = params_dMdh[len(r):2*len(r)]
    return r, eta
                                
