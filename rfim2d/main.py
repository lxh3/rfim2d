from .general_use import load_svA, load_hvdMdh, load_func, save_func
from .fitting import fit_As_Scaling, fit_dMdh_Scaling, joint_fit, divvy_params
from .plotting import plot_xy_and_xfofx
from .scaling import Sigma_func, eta_func 



# Function to perform all of the fits for given data files

def perform_all_fits(filenames=None, data=None, joint=True, sigmaNu_fixed=True, simple=False, save=False, priorWeight=0.1):
    """
    Perform the fit of A, dMdh, Sigma and eta and return values
    Input:
        filenames - [A_filename, dMdh_filename] - filenames to load data from
            A_filename - location where area weighted size distribution is stored
            dMdh_filename - location where dM/dh data is stored
        data - [dataA, datadMdh] - if filenames==None, checks to see if data was
                                   provided here manually
            dataA - [r,s,A,As]
            datadMdh - [r,h,dMdh]
        sigmaNu_fixed - flag for whether to fit the value of sigmaNu or fix it to 0.5
        simple - flag for whether to use the Sigma derived from the simplest dw/dl
        priorWeight - weight used in the fit of eta
    Output:
        params_A - fit values found for A 
        params_dMdh - fit values found for dM/dh
        params_Sigma - fit values found for Sigma
        params_eta - fit values found for eta
    """
    if filenames == None:
        if data==None:
            r,s,A,As = load_svA()
            r,h,dMdh = load_hvdMdh()
        else:
            r,s,A,As = data[0]
            r,h,dMdh = data[1]

    else:
        A_filename, dMdh_filename = filenames
        r,s,A,As = load_func(A_filename)
        r,h,dMdh = load_func(dMdh_filename)

    args_A = [s, As]
    params_A, err = fit_As_Scaling(args_A, save=save)
    Sigma = params_A[:len(r)]

    args_dMdh = [h,dMdh]
    params_dMdh, err = fit_dMdh_Scaling(args_dMdh, save=save)
    eta = params_dMdh[len(r):2*len(r)]

    if joint:
        args = [r, Sigma, eta, priorWeight]
        params, err = joint_fit(args, sigmaNu_fixed=sigmaNu_fixed, simple=simple, save=save)
        params_Sigma, params_eta, prior = divvy_params(params, sigmaNu_fixed, priorWeight=priorWeight)

    else:
        args_Sigma = [r, Sigma]
        params_Sigma, err = Sigma_fit(args, sigmaNu_fixed=sigmaNu_fixed, simple=simple, save=save)
        args_eta = [r, eta, priorWeight]
        params_eta, err = eta_fit(args, simple=simple, save=save)
       
    return params_A, params_dMdh, params_Sigma, params_eta




# Get and plot Sigma and eta

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

def plot_Sigma(r, Sigma, params, labels=[r'$r$',r'$\Sigma(r)$'], logscale=[False,True], simple=False, scaled=False, figure_name=None):
    data = [None, r, Sigma]
    plot_xy_and_xfofx(data, Sigma_func, labels, params, logscale=logscale, simple=simple, scaled=scaled, figure_name=figure_name)
    return

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
        r,h,dMdh = load_hvdMdh()
    else:
        r,h,dMdh = load_func(dMdh_filename)

    args_dMdh = [h, dMdh]
    params_dMdh, err = fit_dMdh_Scaling(args_dMdh)
    eta = params_dMdh[len(r):2*len(r)]
    return r, eta

def plot_eta(r, eta,  params, labels=[r'$r$',r'$\eta(r)$'], logscale=[False,False], simple=False, figure_name=None):
    data = [None, r, eta]
    plot_xy_and_xfofx(data, eta_func, labels, params, logscale=logscale, simple=simple, figure_name=figure_name)
    return

def get_and_plot_Sigma_and_eta(A_filename=None, dMdh_filename=None, joint=True, sigmaNu_fixed=True, simple=False, figure_name=None, save_fits=False):
    r,Sigma = get_Sigma(A_filename)
    r,eta = get_eta(dMdh_filename)
    params_A, params_dMdh, params_Sigma, params_eta = perform_all_fits(A_filename,dMdh_filename,joint=joint,sigmaNu_fixed=sigmaNu_fixed,simple=simple, save=save_fits)
    plot_Sigma(r, Sigma, params_Sigma, simple=simple, figure_name='Sigma_'+figure_name)
    plot_eta(r, eta, params_eta, simple=simple, figure_name='eta_'+figure_name)   
    return [r, Sigma, params_Sigma, eta, params_eta]



# Fit subsets of r

def slice_data(data, start, end):
    rtemp = data[0][start:end]
    if len(data) == 4:
        r,s,A,As = data
        stemp = s[start:end]
        Atemp = A[start:end]
        Astemp = As[start:end]
        datatemp = [rtemp, stemp, Atemp, Astemp]
    else:
        r,h,dMdh = data 
        htemp = h[start:end]
        dMdhtemp = dMdh[start:end]
        datatemp = [rtemp, htemp, dMdhtemp]
    return datatemp

def fit_subsets_of_r(A_filename=None, dMdh_filename=None, num=3, joint=True, simple=False):

    if A_filename == None:
        dataA = load_svA()
    else:
        dataA = load_func(A_filename)

    if dMdh_filename == None:
        datadMdh = load_hvdMdh()
    else:
        datadMdh = load_func(dMdh_filename)
  
    r = dataA[0]  
    params_A = []
    params_dMdh = []
    params_Sigma = []
    params_eta = []

    numCurves = len(r)
    for i in range(numCurves-num+1):
        dataAtemp = slice_data(dataA, i, i+num)
        datadMdhtemp = slice_data(datadMdh, i, i+num)
        pA,pM,pS,pe = perform_all_fits(data=[dataAtemp,datadMdhtemp], joint=joint, simple=simple)
        params_A.append(pA)
        params_dMdh.append(pM)
        params_Sigma.append(pS)
        params_eta.append(pe)

    return params_A, params_dMdh, params_Sigma, params_eta 
