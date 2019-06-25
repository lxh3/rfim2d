from .save_and_load import load_svA, load_hvdMdh
from .fitting import perform_all_fits, get_Sigma, get_eta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.utils import io


default_fixed = dict([('df', 2.), ('C', 0.)])


def slice_data(data, start, end):
    """
    Extract a subsection of the data for a specified range of r
    Input:
        data - either dataA or datadMdh:
            dataA - [r, s, A, As]
            datadMdh - [r, h, dMdh]
        start - starting index for r values to extract
        end - ending index for r values to extract
    Output:
        datatemp - subsection of the data corresponding to r[start:end]
    """
    rtemp = data[0][start:end]
    if len(data) == 4:
        r, s, A, As = data
        stemp = s[start:end]
        Atemp = A[start:end]
        Astemp = As[start:end]
        datatemp = [rtemp, stemp, Atemp, Astemp]
    else:
        r, h, dMdh = data
        htemp = h[start:end]
        dMdhtemp = dMdh[start:end]
        datatemp = [rtemp, htemp, dMdhtemp]
    return datatemp


def fit_subsets_of_r(filenames=[None, None], num=11,
                     fixed_dict=default_fixed,
                     func_type='truncated', verbose=False):
    """
    NOTE: This function currently requires that the simulation
          data for A and dM/dh be obtained from the same set
          of disorder values, r

    Perform the fit of A, dMdh, Sigma and eta for subsets of
    the simulated r values and return the parameters for each fit
    Input:
        filenames - [A_filename, dMdh_filename]
                  - filenames to load data from
            A_filename - location where area weighted size
                         distribution is stored
            dMdh_filename - location where dM/dh data is stored
        num - size of the subsets of r to consider
        func_type - functional form to use for joint fit of Sigma and eta
        verbose - flag to print cost of fits
    Output:
        params_A - list of fit values found for A for each
                   subset of r values considered
        params_dMdh - list of fit values found for dM/dh for each
                   subset of r values considered
        params_Sigma - list of fit values found for Sigma for each
                   subset of r values considered
        params_eta - list of fit values found for eta for each
                   subset of r values considered
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
        pA, pM, pS, pe = perform_all_fits(data=[dataAtemp, datadMdhtemp],
                                          fixed_dict=fixed_dict,
                                          func_type=func_type,
                                          verbose=verbose,
                                          show_params=False)
        params_A.append(list(pA.values()))
        params_dMdh.append(list(pM.values()))
        params_Sigma.append(list(pS.values()))
        params_eta.append(list(pe.values()))

    return params_A, params_dMdh, params_Sigma, params_eta


def std_funcs(r, subset_size, func_values):
    """
    Takes a list of disorders, subset_size, and function values determined 
    for different subsets of r
    Returns the standard deviation of the function value for overlapping
    r values
    """
    # Set up empty dataframe
    columns = [str(r_val) for r_val in r]
    index = [x for x in range(len(r)-subset_size+1)]
    df =  pd.DataFrame(columns=columns, index=index)
    # Fill in dataframe 
    for i in range(len(func_values)):
         r_vals = [ str(r_val) for r_val in r[i:i+subset_size]]
         func_values_dict = dict(zip(r_vals, func_values[i]))
         df.loc[i] = pd.Series(func_values_dict)
    # Create error dict
    border_len = len(r)-subset_size
    subset_r = r[border_len:-border_len]
    errors = np.std(np.asarray(df.dropna(axis=1).values,dtype=np.float32), axis=0)
    error_dict = dict(zip(subset_r, errors))
    return error_dict


def get_func_errors(subset_size, args, filenames):
    """"
    Takes subset size and function values extracted from the fit
    Returns Sigma and eta along with their standard deviations
    """
    pA, pA_list, pdMdh, pdMdh_list = args

    r, s, A, As = load_svA(filenames[0])

    Sigma = dict(zip(r, list(pA['Sigma'])))
    eta = dict(zip(r, list(pdMdh['eta'])))

    Sigma_list = [pA[0] for pA in pA_list]
    eta_list = [pdMdh[1] for pdMdh in pdMdh_list]

    Sigma_std = std_funcs(r, subset_size, Sigma_list)
    eta_std = std_funcs(r, subset_size, eta_list)

    return Sigma, Sigma_std, eta, eta_std 


def std_params(params):
    """
    Takes a list of parameters determined for different subsets of r
    Returns the standard deviation in each parameter
    """
    std_params = []
    for i in range(len(params[0])):
        std_params.append(np.std(np.array(params)[:, i]))
    return std_params


def plot_text(labels, params, std_params, figure_name=None):
    """
    Plot figure with the parameters listed with their
    values +/- errors
    Input:
        labels - parameter names
        params - parameter values
        std_params - parameter errors
        figure_name - if provided, figure is saved under this name
    """
    plt.figure(figsize=(4.25, 2.))
    annotation_string = labels[0]
    annotation_string += r' %.4f $\pm$ %.4f'%(params[0], std_params[0])
    for i in range(len(labels)-1):
        annotation_string += "\n"
        annotation_string += labels[i+1]
        annotation_string += (r' %.4f $\pm$ '
                              r'%.4f'%(params[i+1], std_params[i+1]))
    plt.annotate(annotation_string, xy=(0.1, 0.1), fontsize=14)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if figure_name is not None:
        plt.savefig(figure_name, bbox_inches='tight')
    plt.show()
    plt.close()
    return


def fit_and_plot_errors(filenames=[None, None], num=11,
                        fixed_dict=default_fixed, func_type='truncated',
                        figure_names=None, verbose=False):
    """
    Perform all fits, determine errors, and plot figures of
    the parameters listed with their values +/- errors
    for the functions Sigma and eta
    Input:
        filenames - location to load simulation data from if not default
        num - number of points in each fit subset used to determine
              parameter errors
        fixed_dict - fixed values in the fit of Sigma and eta
        func_type - functional form for dw/dl used to determine
                    Sigma and eta
        figure_names - if provided, figures are saved under these name
        verbose - flag to print cost of fits
    Output:
        func_data - errors in the functional fit (A and dMdh)
        param_data - pS, params_Sigma_std, peta, params_eta_std
            pS - best fit parameters for Sigma
            params_Sigma_std - errors associated with the best fit
                               parameters for Sigma
            peta - best fit parameters for eta
            params_eta_std - errors associated with the best fit
                             parameters for eta
    """
    # Perform all fitting procedures
    with io.capture_output() as captured:
        pA, pdMdh, pS, peta = perform_all_fits(filenames=filenames,
                                               fixed_dict=fixed_dict,
                                               func_type=func_type,
                                               verbose=verbose,
                                               show_params=False)
        lists = fit_subsets_of_r(num=num,
                                 fixed_dict=fixed_dict,
                                 func_type=func_type,
                                 verbose=verbose)
        pA_list, pdMdh_list, pS_list, peta_list = lists

    # Calculate errors for functional values (Sigma and eta)
    args = [pA, pA_list, pdMdh, pdMdh_list] 
    func_data = get_func_errors(num, args, filenames)

    # Calculate errors for parameter values
    params_Sigma_std = std_params(pS_list)
    params_eta_std = std_params(peta_list)

    param_data = pS, params_Sigma_std, peta, params_eta_std

    # Plot parameter values with the associated standard deviation 
    labels_Sigma = list(pS.keys())
    labels_eta = list(peta.keys())
    params_Sigma = list(pS.values())
    params_eta = list(peta.values())


    if figure_names is not None:
        plot_text(labels_Sigma, params_Sigma, params_Sigma_std,
                  figure_name=figure_names[0])
        plot_text(labels_eta, params_eta, params_eta_std,
                  figure_name=figure_names[1])

    return func_data, param_data


def get_function_errors(func_type='truncated'):
    """
    Get errors for Sigma and eta (used to produce Figure 3: 
    see plotting.py 'plot_Sigma_compare_with_eta_inset')
    """

    r,Sigma = get_Sigma()
    r,eta = get_eta()

    func, params = fit_and_plot_errors(func_type=func_type)

    #Sigma
    r_small = list(func[1].keys())
    Sig_vals = [func[0][i] for i in r_small]
    err = list(func[1].values())
    Sigma_errors = [r_small, Sig_vals, err]

    #eta
    r_small = list(func[3].keys())
    eta_vals = [func[2][i] for i in r_small]
    err = list(func[3].values())
    eta_errors = [r_small, eta_vals, err]     # Calculate error bars

    return Sigma_errors, eta_errors

