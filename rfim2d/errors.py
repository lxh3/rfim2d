import numpy as np
import matplotlib.pyplot as plt
from IPython.utils import io

from .save_and_load import load_svA, load_hvdMdh, load_func
from .fitting import fit_As_Scaling, fit_dMdh_Scaling, joint_fit, perform_all_fits
from .param_dict import divvy_params

def slice_data(data, start, end):
    """
    Extract a subsection of the data for a specified range of r
    Input:
        data - either dataA or datadMdh:
            dataA - [r,s,A,As]
            datadMdh - [r,h,dMdh]
        start - starting index for r values to extract
        end - ending index for r values to extract
    Output:
        datatemp - subsection of the data corresponding to r[start:end]
    """
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


def fit_subsets_of_r(filenames=[None,None], num=11, func_type='wellbehaved', verbose=False):
    """
    NOTE: This function currently requires that the simulation data for A and dM/dh
          be obtained from the same set of disorder values, r

    Perform the fit of A, dMdh, Sigma and eta for subsets of the simulated r values 
    and return the parameters for each fit
    Input:
        filenames - [A_filename, dMdh_filename] - filenames to load data from
            A_filename - location where area weighted size distribution is stored
            dMdh_filename - location where dM/dh data is stored
        num - size of the subsets of r to consider
        func_type - functional form to use for joint fit of Sigma and eta
        verbose - flag to print cost of fits
    Output:
        params_A - list of fit values found for A for each subset of r values considered
        params_dMdh - list of fit values found for dM/dh for each subset of r values considered
        params_Sigma - list of fit values found for Sigma for each subset of r values considered
        params_eta - list of fit values found for eta for each subset of r values considered
    """
    dataA = load_svA(filenames[0])
    datadMdh = load_hvdMdh(filenames[1])
  
    r = dataA[0]
    params_A = []
    params_dMdh = []
    params_Sigma = []
    params_eta = []

    numCurves = len(r)
    for i in range(numCurves-num+1):
        dataAtemp = slice_data(dataA, i, i+num)
        datadMdhtemp = slice_data(datadMdh, i, i+num)
        pA,pM,pS,pe = perform_all_fits(data=[dataAtemp,datadMdhtemp], func_type=func_type, verbose=verbose, show_params=False)
        params_A.append(list(pA.values()))
        params_dMdh.append(list(pM.values()))
        params_Sigma.append(list(pS.values()))
        params_eta.append(list(pe.values()))

    return params_A, params_dMdh, params_Sigma, params_eta

def std_params(params):
    """
    Takes a list of parameters determined for different subsets of r
    Returns the standard deviation in each parameter
    """
    std_params = []
    for i in range(len(params[0])):
        std_params.append(np.std(np.array(params)[:,i]))
    return std_params

def plot_text(labels,params,std_params,figure_name=None):
    """
    Plot figure with the parameters listed with their 
    values +/- errors
    Input:
        labels - parameter names
        params - parameter values
        std_params - parameter errors
        figure_name - if provided, figure is saved under this name
    """
    fig = plt.figure(figsize=(4.25,1.75))
    annotation_string = labels[0]
    annotation_string += r' %.4f $\pm$ %.4f'%(params[0],std_params[0])
    for i in range(len(labels)-1):
        annotation_string += "\n"
        annotation_string += labels[i+1]
        annotation_string += r' %.4f $\pm$ %.4f'%(params[i+1],std_params[i+1])
    plt.annotate(annotation_string, xy=(0.1, 0.1),fontsize=14)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if figure_name != None:
        plt.savefig(figure_name, bbox_inches='tight')
    plt.show()
    plt.close()
    return

def fit_and_plot_errors(filenames=[None,None], num=11, func_type='wellbehaved',figure_names=None, verbose=False):
    """
    Perform all fits, determine errors, and plot figures of 
    the parameters listed with their values +/- errors
    for the functions Sigma and eta
    Input:
        filenames - location to load simulation data from if not default
        num - number of points in each fit subset used to determine parameter errors
        func_type - functional form for dw/dl used to determine Sigma and eta
        figure_names - if provided, figures are saved under these name
        verbose - flag to print cost of fits
    Output:
        params_Sigma - best fit parameters for Sigma
        params_Sigma_std - errors associated with the best fit parameters for Sigma
        params_eta - best fit parameters for eta
        params_eta_std - errors associated with the best fit parameters for eta
    """
    with io.capture_output() as captured:
        params_A,params_dMdh,params_Sigma,params_eta = perform_all_fits(filenames=filenames, func_type=func_type, verbose=verbose, show_params=False)
        params_A_list,params_dMdh_list,params_Sigma_list,params_eta_list = fit_subsets_of_r(num=num, func_type=func_type, verbose=verbose)

    labels_Sigma = list(params_Sigma.keys())
    labels_eta = list(params_eta.keys())
    params_Sigma = list(params_Sigma.values())
    params_eta = list(params_eta.values())

    params_Sigma_std = std_params(params_Sigma_list)
    params_eta_std = std_params(params_eta_list)

    if figure_names != None:
        plot_text(labels_Sigma, params_Sigma, params_Sigma_std, figure_name=figure_names[0])
        plot_text(labels_eta, params_eta, params_eta_std, figure_name=figure_names[1])

    return params_Sigma, params_Sigma_std, params_eta, params_eta_std
