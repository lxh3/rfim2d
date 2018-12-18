import matplotlib.pyplot as plt
import numpy as np

from . import save_and_load, scaling, fitting, param_dict
from .scaling import Sigma_func, eta_func
from .param_dict import split_dict


#### SUPPORTING FUNCTIONS ####

def parse_kwargs(kwargs, a_list):
    """
    extract values from kwargs or set default
    """
    if a_list != None:
        num_colors = len(a_list)
        default_colors = generate_colors(num_colors)
    else:
        num_colors = 1
        default_colors = 'k'

    logscale = kwargs.get('logscale',[False,False])
    Range = kwargs.get('Range', [[],[]])
    colors = kwargs.get('colors', default_colors)
    figure_name = kwargs.get('figure_name',None)
    show = kwargs.get('show',True)

    values = [logscale, Range, colors, figure_name, show]

    return values

def generate_colors(num_colors):
    """
    Randomly generates color
    Input:
        num_colors - number of randomly generated colors requested
    Output:
        color_list - list of size [num_colors, 3] containing the RBG values for each color requested
    """
    c = np.random.rand(3,num_colors)
    color_list = c.T
    return color_list

def generate_points(x_list, function, constant, func_type, minmax=None, scaled=False):
    """
    generate (r,Sigma(r)) points to plot function behavior at 
    points intermediate to those simulated
    """
    if minmax == None:
        minimum_r = np.min(x_list)
        maximum_r = np.max(x_list)
    else:
        minimum_r = minmax[0]
        maximum_r = minmax[1]
    step_size = (maximum_r-minimum_r)/100.
    r_vals = np.arange(minimum_r, maximum_r, step_size)
    funcr_vals = function(r_vals, constant, func_type=func_type, scaled=scaled)
    return r_vals, funcr_vals


#### PLOT SETUP/FINISH ####

def setup_plot(labels, logscale, Range):
    """
    Setup a plot with the appropriate, scale, axes and labels
    on which the data will be displayed
    """
    # Initialize Plot
    plt.figure(figsize=(10.,5.*(np.sqrt(5.)-1)))
    ax = plt.subplot(111)

    # Set logscale if applicable
    if logscale[0]:
        ax.set_xscale('log')
    if logscale[1]:
        ax.set_yscale('log')

    # Set Range if applicable
    if len(Range[0])==2:
        plt.xlim(Range[0])
    if len(Range[1])==2:
        plt.ylim(Range[1])

    # Labels 
    plt.xlabel(labels[0],fontsize=40)
    plt.ylabel(labels[1],fontsize=40)

    return ax


def finish_plot(ax, figure_name, loc='upper left', legend_font_size=12, show=True):
    """
    Add a legend to the plot, save and close
    """
    # Legend
    legend = ax.legend(loc=loc,  shadow=True, fontsize=legend_font_size)
    # Make sure labels fully visible
    plt.tight_layout()
    # Save figure and close plot 
    if figure_name != None:
        plt.savefig(figure_name,bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return




#### MAIN PLOTTING FUNCTIONS ####

def scatter(data, labels, **kwargs):
    """
    Plot x vs y for each value of r
    (Used to plot raw data of A and dM/dh)
    """
    r_list, x_list, y_list = data

    logscale, Range, colors, figure_name, show = parse_kwargs(kwargs, r_list)

    ax = setup_plot(labels, logscale, Range)

    # For each value of r, plot x vs y with a corresponding color
    for r, x, y, color in zip(r_list, x_list, y_list, colors):
        ax.scatter(x, y, c=color, label=r, lw=0)

    finish_plot(ax, figure_name, show=show)

    return


def scatter_vs_function(data, function, labels, constant, constant_given_r, **kwargs):
    """
    Plot x vs y and x vs function(x) for each value of r 
    (Used to plot scatter vs fit of A and dM/dh)
    """
    r_list, x_list, y_list = data
   
    logscale, Range, colors, figure_name, show = parse_kwargs(kwargs, r_list)

    ax = setup_plot(labels, logscale, Range)

    for r, x, y, cgr, color in zip(r_list, x_list, y_list, constant_given_r, colors):
        if isinstance(cgr, dict):
            key, cgr = split_dict(cgr)
        ax.scatter(x, y, color=color,label=str(r),lw=0)
        ax.plot(x,function(x, cgr, constant),color=color)
        loc='upper left'

    finish_plot(ax, figure_name, loc, show=show)

    return


def collapse(data, function, labels, constant, constant_given_r, **kwargs):
    """
    Plot the scaling collapse of the data (x,y) and fitting function output (x,function(y))  
    (Used to plot the scaling collapse of A and dM/dh)
    """
    r_list, x_list, y_list = data

    logscale, Range, colors, figure_name, show = parse_kwargs(kwargs, r_list)

    ax = setup_plot(labels, logscale, Range)

    for r, x, y, cgr, color in zip(r_list, x_list, y_list, constant_given_r, colors):
        if isinstance(cgr, dict):
            key, cgr = split_dict(cgr)
        xscaled, yscaled, yscaled_from_function = function(x, y, cgr, constant)
        ax.scatter(xscaled, yscaled, color=color,label=str(r), lw=0)
        ax.plot(xscaled, yscaled_from_function, color='black', lw=2)

    finish_plot(ax, figure_name, show=show)

    return


def compare(data, function, labels, constant, **kwargs):
    """
    Create a scatter plot of the data (x,y) and the best fit using each of the functional forms (x,function(y))  
    (Used to plot Sigma or eta data compared with the best fits using different functional forms)
    """
    x_list, y_list = data

    if not isinstance(constant[0],list):
        constant = [constant]

    types = kwargs.get('types', ['powerlaw', 'simple', 'wellbehaved', 'pitchfork'])
    if isinstance(types, str):
        types = [types]

    loc = kwargs.get('loc', 'upper left')

    logscale, Range, colors, figure_name, show = parse_kwargs(kwargs, types)

    ax = setup_plot(labels, logscale, Range)
    ax.scatter(x_list, y_list, color='k', label='data')

    scaled = kwargs.get('scaled', False)
    minmax = kwargs.get('minmax', None)
    for i, t in enumerate(types):
        x, y = generate_points(x_list, function, constant[i], t, minmax=minmax, scaled=scaled)
        ax.plot(x, y, color=colors[i], lw=2, label=t)

    finish_plot(ax, figure_name, loc=loc, legend_font_size=20, show=show)

    return



#### Get and plot Sigma and eta ####

def plot_Sigma(r, Sigma, params, func_type, scaled=False, figure_name='Sigma_fit.png'):
    """
    Plot default Sigma plot using compare function
    """
    ls = [False, True]
    loc = 'upper right'
    labels = [r'$r$',r'$\Sigma(r)$']
    compare([r,Sigma],scaling.Sigma_func,labels,params,logscale=ls,loc=loc,types=[func_type],figure_name=figure_name,scaled=scaled)
    return 

def plot_eta(r, eta, params, func_type, scaled=False, figure_name='eta_fit.png'):
    """
    Plot default eta plot using compare function
    """
    ls = [False, True]
    loc = 'lower right'
    labels = [r'$r$',r'$\eta(r)$']
    compare([r,eta],scaling.eta_func,labels,params,logscale=ls,loc=loc,types=[func_type],figure_name=figure_name,scaled=scaled)
    return

def get_and_plot_Sigma_and_eta(filenames=[None,None], sigmaNu_fixed=True, func_type='wellbehaved', figure_names=[None,None]):
    """
    Perform A and dM/dh fits to determine Sigma(r) and eta(r)
    then perform the fits of the Sigma(r) and eta(r) functional forms 
    to create the default plots for each using the compare function
    """
    data_Sigma = fitting.get_Sigma(filenames[0])
    data_eta = fitting.get_eta(filenames[1])
    ft = func_type
    sNf = sigmaNu_fixed
    params_A, params_dMdh, params_Sigma, params_eta = fitting.perform_all_fits(filenames, sigmaNu_fixed=sNf, func_type=ft, show_params=False)

    ls = [False, True]
    # Plot Sigma
    loc = 'upper right'
    labels = [r'$r$',r'$\Sigma(r)$']
    params_Sigma_vals = list(params_Sigma.values())
    compare(data_Sigma, Sigma_func, labels, params_Sigma_vals, logscale=ls, loc=loc, types=[func_type], figure_name=figure_names[0])
    # Plot eta
    loc = 'lower right'
    labels = [r'$r$',r'$\eta(r)$']
    params_eta_vals = list(params_eta.values())
    compare(data_eta, eta_func, labels, params_eta_vals, logscale=ls, loc=loc, types=[func_type], figure_name=figure_names[1])
    return data_Sigma, params_Sigma, data_eta, params_eta

