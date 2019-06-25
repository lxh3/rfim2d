from . import scaling, fitting
from .scaling import Sigma_func, eta_func
from .param_dict import split_dict, separate_params
from .errors import fit_and_plot_errors, get_function_errors
from .residuals import linear_residual, linear_function

import matplotlib.pyplot as plt
import numpy as np


default_fixed = dict([('df', 2.), ('C', 0.)])


# SUPPORTING FUNCTIONS

def parse_kwargs(kwargs, a_list):
    """
    extract values from kwargs or set default
    """
    if a_list is not None:
        num_colors = len(a_list)
        default_colors = generate_colors(num_colors)
    else:
        num_colors = 1
        default_colors = 'k'

    logscale = kwargs.get('logscale', [False, False])
    Range = kwargs.get('Range', [[], []])
    colors = kwargs.get('colors', default_colors)
    figure_name = kwargs.get('figure_name', None)
    show = kwargs.get('show', True)
    dist = kwargs.get('dist', None)

    values = [logscale, Range, colors, figure_name, show, dist]

    return values


def generate_colors(num_colors):
    """
    Randomly generates color
    Input:
        num_colors - number of randomly generated colors requested
    Output:
        color_list - list of size [num_colors, 3] containing the
                     RBG values for each color requested
    """
    c = np.random.rand(3, num_colors)
    color_list = c.T
    return color_list


def generate_points(x_list, function, constant, func_type,
                    minmax=None, scaled=False):
    """
    generate (r,Sigma(r)) points to plot function behavior at
    points intermediate to those simulated
    """
    if minmax is None:
        minimum_r = np.min(x_list)
        maximum_r = np.max(x_list)
    else:
        minimum_r = minmax[0]
        maximum_r = minmax[1]
    step_size = (maximum_r-minimum_r)/100.
    r_vals = np.arange(minimum_r, maximum_r, step_size)
    funcr_vals = function(r_vals, constant,
                          func_type=func_type, scaled=scaled)
    return r_vals, funcr_vals


# PLOT SETUP/FINISH

def setup_plot(labels, logscale, Range):
    """
    Setup a plot with the appropriate, scale, axes and labels
    on which the data will be displayed
    """
    # Initialize Plot
    fig = plt.figure(figsize=(10., 5.*(np.sqrt(5.)-1)))
    ax = plt.subplot(111)

    # Set logscale if applicable
    if logscale[0]:
        ax.set_xscale('log')
    if logscale[1]:
        ax.set_yscale('log')

    # Set Range if applicable
    if len(Range[0]) == 2:
        plt.xlim(Range[0])
    if len(Range[1]) == 2:
        plt.ylim(Range[1])

    # Set tick label size
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)

    # Labels
    plt.xlabel(labels[0], fontsize=40, labelpad=10)
    plt.ylabel(labels[1], fontsize=40, labelpad=10)

    return fig, ax


def finish_plot(ax, figure_name, loc='upper left',
                legend_font_size=16, show=True):
    """
    Add a legend to the plot, save and close
    """
    # Legend
    ax.legend(loc=loc, shadow=True, fontsize=legend_font_size)
    # Make sure labels fully visible
    plt.tight_layout()
    # Save figure and close plot
    if figure_name is not None:
        plt.savefig(figure_name, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return


# MAIN PLOTTING FUNCTIONS

def scatter(data, labels, **kwargs):
    """
    Plot x vs y for each value of r
    (Used to plot raw data of A and dM/dh)
    """
    r_list, x_list, y_list = data

    kwarg_list = parse_kwargs(kwargs, r_list)
    logscale, Range, colors, figure_name, show, dist = kwarg_list

    fig, ax = setup_plot(labels, logscale, Range)

    # For each value of r, plot x vs y with a corresponding color
    for x, y, color in zip(x_list, y_list, colors):
        ax.scatter(x, y, c=color, lw=0)

    # Make legend points larger
    for r, color in zip(r_list, colors):
        plt.scatter([], [], c=color, s=50, label=r)

    finish_plot(ax, figure_name, show=show)

    return


def scatter_vs_function(data, function, labels, constant,
                        constant_given_r, **kwargs):
    """
    Plot x vs y and x vs function(x) for each value of r
    (Used to plot scatter vs fit of A and dM/dh)
    """
    r_list, x_list, y_list = data

    if isinstance(constant, dict):
        keys, constant = split_dict(constant)

    kwarg_list = parse_kwargs(kwargs, r_list)
    logscale, Range, colors, figure_name, show, dist = kwarg_list
    fig, ax = setup_plot(labels, logscale, Range)

    zipped = zip(r_list, x_list, y_list, constant_given_r, colors)
    for r, x, y, cgr, color in zipped:
        if isinstance(cgr, dict):
            key, cgr = split_dict(cgr)
        ax.scatter(x, y, color=color, label=str(r), lw=0)
        ax.plot(x, function(x, cgr, constant), color=color)
        loc = 'upper left'

    finish_plot(ax, figure_name, loc, show=show)

    return


def collapse(data, function, labels, constant, constant_given_r, **kwargs):
    """
    Plot the scaling collapse of the data (x,y) and fitting function
    output (x,function(y)) (Used to plot the scaling collapse
    of A and dM/dh)
    """
    r_list, x_list, y_list = data

    if isinstance(constant, dict):
        keys, constant = split_dict(constant)

    kwarg_list = parse_kwargs(kwargs, r_list)
    logscale, Range, colors, figure_name, show, dist = kwarg_list

    fig, ax = setup_plot(labels, logscale, Range)

    zipped = zip(r_list, x_list, y_list, constant_given_r, colors)
    for r, x, y, cgr, color in zipped:
        if isinstance(cgr, dict):
            key, cgr = split_dict(cgr)
        x, y, y_from_function = function(x, y, cgr, constant)
        ax.scatter(x, y, color=color, label=str(r), lw=0)
        #if r == r_list[0]:
        #    ax.plot(x, y_from_function, color='black', lw=2)

    finish_plot(ax, figure_name, show=show)

    return


def collapse_with_inset(data, function, labels, constant, constant_given_r, **kwargs):
    """
    Plot the scaling collapse of the data (x,y) and fitting function
    output (x,function(y)) (Used to plot the scaling collapse
    of A and dM/dh) - contains inset with the raw data
    """
    r_list, x_list, y_list = data

    if isinstance(constant, dict):
        keys, constant = split_dict(constant)

    kwarg_list = parse_kwargs(kwargs, r_list)
    logscale, Range, colors, figure_name, show, dist = kwarg_list

    fig, ax = setup_plot(labels, logscale, Range)

    zipped = zip(r_list, x_list, y_list, constant_given_r, colors)
    for r, x, y, cgr, color in zipped:
        if isinstance(cgr, dict):
            key, cgr = split_dict(cgr)
        x, y, y_from_function = function(x, y, cgr, constant)
        ax.scatter(x, y, color=color, label=str(r), lw=0)

    if dist == 'A':
        Range2 = [[1e0,1e5],[1e-4,0.75e0]]
    elif dist == 'dMdh':
        Range2 = [[-3,4], [1e-2,30]]
    else:
        Range2 = None

    inset1 = fig.add_axes([.175, .65, .3, .3])
    for i in range(len(r_list)):
        inset1.scatter(x_list[i], y_list[i], color=colors[i], s=10)
        if Range2 is not None:
            inset1.set_xlim(Range2[0])
            inset1.set_ylim(Range2[1])

    # Set logscale if applicable
    if logscale[0]:
        xscale = 'log'
    else:
        xscale = 'linear'

    plt.setp(inset1, xticks=[], yticks=[], xscale=xscale, yscale='log')
    plt.setp(inset1.get_xticklabels(), visible=False);
    plt.setp(inset1.get_yticklabels(), visible=False);

    # Make sure labels fully visible
    plt.tight_layout()
    # Save figure and close plot
    if figure_name is not None:
        plt.savefig(figure_name, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    return


def compare(data, function, labels, constant, errors=None, **kwargs):
    """
    Create a scatter plot of the data (x,y) and the best fit
    using each of the functional forms (x,function(y))
    (Used to plot Sigma or eta data compared with the best
    fits using different functional forms)
    """
    x_list, y_list = data

    types = kwargs.get('types', ['power law', 'truncated',
                                 'well-behaved', 'pitchfork'])
    linetypes = kwargs.get('linetypes',['--', '-', '-.', ':'])

    if isinstance(types, str):
        types = [types]
        constant = [constant]

    loc = kwargs.get('loc', 'upper left')

    kwarg_list = parse_kwargs(kwargs, types)
    logscale, Range, colors, figure_name, show, dist = kwarg_list

    fig, ax = setup_plot(labels, logscale, Range)
    ax.scatter(x_list, y_list, s=50, color='k', label='data')

    if errors is not None:
        r, func_val, err = errors
        ax.errorbar(r, func_val, yerr = err, color='k', ls='none')

    scaled = kwargs.get('scaled', False)
    minmax = kwargs.get('minmax', None)
    for i, t in enumerate(types):
        x, y = generate_points(x_list, function, constant[i], t,
                               minmax=minmax, scaled=scaled)
        if len(types) == 1:
            ax.plot(x, y, color=colors[i], lw=2, label=labels[1])
        else:
            ax.plot(x, y, linetypes[i], color=colors[i], lw=2, label=t)

    finish_plot(ax, figure_name, loc=loc, legend_font_size=20, show=show)

    return


# Get and plot Sigma and eta

def plot_Sigma(r, Sigma, params, func_type, errors=None, 
               scaled=False, figure_name='Sigma_fit.png'):
    """
    Plot default Sigma plot using compare function
    """
    ls = [False, True]
    loc = 'upper right'
    labels = [r'$r$', r'$\Sigma(r)$']
    compare([r, Sigma], scaling.Sigma_func, labels, [params],
            errors=errors, logscale=ls, loc=loc, types=[func_type],
            figure_name=figure_name, scaled=scaled)
    return


def plot_eta(r, eta, params, func_type, errors=None, 
             scaled=False, figure_name='eta_fit.png'):
    """
    Plot default eta plot using compare function
    """
    ls = [False, True]
    loc = 'lower right'
    labels = [r'$r$', r'$\eta(r)$']
    compare([r, eta], scaling.eta_func, labels, [params],
            errors=errors, logscale=ls, loc=loc, types=[func_type],
            figure_name=figure_name, scaled=scaled)
    return


def get_and_plot_Sigma_and_eta(filenames=[None, None],
                               df_fixed=default_fixed,
                               func_type='truncated',
                               figure_names=[None, None]):
    """
    Perform A and dM/dh fits to determine Sigma(r) and eta(r)
    then perform the fits of the Sigma(r) and eta(r) functional forms
    to create the default plots for each using the compare function
    """
    data_Sigma = fitting.get_Sigma(filenames[0])
    data_eta = fitting.get_eta(filenames[1])

    func, params = fit_and_plot_errors(filenames=filenames, 
                                       fixed_dict=df_fixed, 
                                       func_type=func_type)
    params_Sigma = params[0]
    params_eta = params[2]

    r_errors = list(func[1].keys())
    Sig_vals_errors = [func[0][i] for i in r_errors]
    err = list(func[1].values())
    Sigma_errors = [r_errors, Sig_vals_errors, err]

    r_errors = list(func[3].keys())
    eta_vals_errors = [func[2][i] for i in r_errors]
    err = list(func[3].values())
    eta_errors = [r_errors, eta_vals_errors, err]

    ls = [False, True]
    # Plot Sigma
    loc = 'upper right'
    labels = [r'$r$', r'$\Sigma(r)$']
    compare(data_Sigma, Sigma_func, labels, [params_Sigma],
            errors=Sigma_errors, logscale=ls, loc=loc, 
            types=[func_type], figure_name=figure_names[0])
    # Plot eta
    loc = 'lower right'
    labels = [r'$r$', r'$\eta(r)$']
    compare(data_eta, eta_func, labels, [params_eta],
            errors=eta_errors, logscale=ls, loc=loc, 
            types=[func_type], figure_name=figure_names[1])

    data = [data_Sigma, params_Sigma, Sigma_errors, 
            data_eta, params_eta, eta_errors]

    return data


def plot_Sigma_compare_with_eta_inset(figure_name=None):
    """Creates Figure 3 in the main text"""

    # Get r, Sigma and eta values
    r,Sigma = fitting.get_Sigma()
    r,eta = fitting.get_eta()

    # Calculate error bars
    Sigma_errors, eta_errors = get_function_errors()

    # Perform each fit
    params_powerlaw, params_pitchfork, params_truncated = fitting.compare_fits(r, Sigma, eta)
    pS_power, pe_power =  separate_params(params_powerlaw, func_type='power law')
    pS_pitch, pe_pitch =  separate_params(params_pitchfork, func_type='pitchfork')
    pS_trun, pe_trun =  separate_params(params_truncated, func_type='truncated')

    # Create main plot (Sigma)
    rlist = np.arange(0.8,8,(8-0.8)/1000)

    labels = [r'$r$', r'$\Sigma(r)$']
    logscale = [False, True]
    Range = [[], []]
    fig, ax = setup_plot(labels, logscale, Range)

    plt.scatter(r, Sigma, c='k', label=r'Data')
    plt.plot(rlist, Sigma_func(rlist, pS_trun, func_type='truncated'),c='b', label=r'NF')
    plt.plot(rlist, Sigma_func(rlist, pS_power, func_type='power law'),c='g', label=r'Power Law')
    plt.plot(rlist, Sigma_func(rlist, pS_pitch, func_type='pitchfork'),c='y', label=r'Pitchfork')

    r_errors, Sigma_vals_errors, err = Sigma_errors
    ax.errorbar(r_errors, Sigma_vals_errors, yerr = err, color='k', ls='none')

    # Create legend
    ax.legend(loc='upper right', shadow=True, fontsize=20)

    # Create inset (eta)
    inset1 = fig.add_axes([.3, .5, .35, .35])
    inset1.scatter(r, eta,c='k',label=r'$\eta_{data}(r)$')
    inset1.plot(rlist, eta_func(rlist, pe_trun, func_type='truncated'),c='b')
    inset1.plot(rlist, eta_func(rlist, pe_power, func_type='power law'),c='g')
    inset1.plot(rlist, eta_func(rlist, pe_pitch, func_type='pitchfork'),c='y')
    r_errors, eta_vals_errors, err = eta_errors
    inset1.errorbar(r_errors, eta_vals_errors, yerr = err, color='k', ls='none')
    inset1.set_xlabel(r'$r$',fontsize=20, labelpad=10)
    inset1.set_ylabel(r'$\eta(r)$',fontsize=20, labelpad=10)
    plt.setp(inset1, yscale='log')
    
    # Save figure
    if figure_name is None:
        plt.savefig('comparison.png',bbox_inches='tight')
    else:
        plt.savefig(figure_name, bbox_inches='tight')

    plt.show()
    plt.close()

    return


def plot_logplot(figure_name=None):
    """Creates Figure 4 in the main text"""

    # Get r, Sigma and eta values
    r,Sigma = fitting.get_Sigma()
    r,eta = fitting.get_eta()

    # Calculate error bars
    Sigma_errors, eta_errors = get_function_errors()

    # Perform each fit
    params_powerlaw, params_pitchfork, params_truncated = fitting.compare_fits(r, Sigma, eta)
    pS_power, pe_power =  separate_params(params_powerlaw, func_type='power law')
    pS_pitch, pe_pitch =  separate_params(params_pitchfork, func_type='pitchfork')
    pS_trun, pe_trun =  separate_params(params_truncated, func_type='truncated')

    args = [r[:7], pS_trun]
    params, err = fitting.perform_fit(linear_residual, [1.,1.], args)

    # Create Plot
    rlist = np.arange(-0.50,8,(8+0.05)/10000)
    plt.figure(figsize=(10., 5.*(np.sqrt(5.)-1)))
    ax = plt.subplot(111)

    plt.scatter(r, 1./np.log(Sigma),c='k',label=r'Data')
    plt.plot(rlist, linear_function(np.asarray(rlist),params),c='r',label='Linear Fit')
    plt.plot(rlist, 1./np.log(Sigma_func(rlist, pS_trun, func_type='truncated')),c='b', label=r'NF')
    plt.plot(rlist, 1./np.log(Sigma_func(rlist, pS_power, func_type='power law')),c='g', label=r'Power Law')
    plt.plot(rlist, 1./np.log(Sigma_func(rlist, pS_pitch, func_type='pitchfork')),c='y', label=r'Pitchfork')
    r_errors, Sigma_vals_errors, err = Sigma_errors
    ax.errorbar(r_errors, Sigma_vals_errors, yerr = err, color='k', ls='none')

    plt.ylim(0.,0.25);
    plt.xlim(-0.5,2.0);

    ax.axvline(x=0.0, ymin=0.0, ymax=1.0, color='k', linestyle='--')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.xlabel(r'$r$', fontsize=30, labelpad=10)
    plt.ylabel(r'$1/\log\Sigma(r)$', fontsize=30, labelpad=10)

    loc='lower right'
    legend_font_size=20
    ax.legend(loc=loc, shadow=True, fontsize=legend_font_size)

    # Save figure
    if figure_name is None:
        plt.savefig('logplot.png',bbox_inches='tight')
    else:
        plt.savefig(figure_name, bbox_inches='tight')

    plt.show()
    plt.close()

    return


def compare_plots_supplement():
    """ Creates the fit comparison plots in the supplement """

    r,Sigma = fitting.get_Sigma()
    r,eta = fitting.get_eta()

    params0 = dict([('rScale',1.0), ('rc', 0.0), ('sScale', 10.),
                   ('etaScale', 1.0), ('df', 2.0), ('lambdaH', 1.0),
                   ('B', 1.0), ('C', 0.0), ('F', 1.0)])

    # Truncated
    fixed_dicts = [dict([('df',2.), ('B',0.)]),
                   dict([('df',2.), ('C',0.)]),
                   dict([('df',2.), ('C',0.), ('rc',0.)]),
                   dict([('df',2.), ('rc',0.)]),
                   dict([('df',2.), ('lambdaH',1.0)]),
                   dict([('df',2.), ('C',0.), ('lambdaH',1.0)]),
                   dict([('df',2.), ('C',0.), ('F',0.), ('lambdaH',1.0)]),
                   dict([('df',2.), ('B',0.), ('C',0.), ('F',0.), ('lambdaH',1.0)])]
    label_list = [r'$d_f=2$, $B=0$',
                  r'$d_f=2$, $C=0$',
                  r'$d_f=2$, $C=0$, $r_c=0$',
                  r'$d_f=2$, $r_c=0$',
                  r'$d_f=2$, $\lambda_h=1$',
                  r'$d_f=2$, $C=0$, $\lambda_h=1$',
                  r'$d_f=2$, $C=0$, $F=0$, $\lambda_h=1$',
                  r'$d_f=2$, $B=0$, $C=0$, $F=0$, $\lambda_h=1$']
    pS_list = []
    pe_list = []

    for fixed_dict in fixed_dicts:
        params, err = fitting.joint_fit([r,Sigma,r,eta], fixed_dict=fixed_dict,
                                        func_type='truncated', params0=params0,
                                        verbose=False,show_params=False)
        pS, pe =  separate_params(params, func_type='truncated')
        pS_list.append(pS)
        pe_list.append(pe)

    Sigma_errors, eta_errors = get_function_errors()
    r_errors, Sigma_vals_errors, err = Sigma_errors
    r_errors, eta_vals_errors, err = eta_errors

    labels = [r'$r$',r'$\Sigma(r)$']
    logscale = [False,True]
    Range = [[], []]
    fig, ax = setup_plot(labels, logscale, Range)
    ax.scatter(r,Sigma,label='data',color='k',s=40)
    ax.errorbar(r_errors, Sigma_vals_errors, yerr = err, color='k', ls='none')
    rlist = np.arange(0.5,10,(10.-0.5)/1000)
    for i, label in enumerate(label_list):
        ax.plot(rlist, Sigma_func(rlist, pS_list[i], func_type='truncated'),label=label)
    figure_name='Sigma_comparison_truncated.png'
    loc = 'upper right'
    plt.xlim([0.5,9.])
    plt.ylim([0.,10**5.])
    finish_plot(ax, figure_name, loc=loc)

    labels = [r'$r$',r'$\eta(r)$']
    logscale = [False,True]
    Range = [[], []]
    fig, ax = setup_plot(labels, logscale, Range)
    ax.scatter(r,eta,label='data',color='k',s=40)
    ax.errorbar(r_errors, eta_vals_errors, yerr = err, color='k', ls='none')
    rlist = np.arange(0.5,10,(10.-0.5)/1000)
    for i, label in enumerate(label_list):
        ax.plot(rlist, eta_func(rlist, pe_list[i], func_type='truncated'),label=label)
    loc = 'lower right'
    figure_name = 'eta_comparison_truncated.png'
    plt.xlim([0.5,9.])
    plt.ylim([10**-2.,10**1.1])
    finish_plot(ax, figure_name, loc=loc)

    # Well-Behaved
    fixed_dicts = [dict([('df',2.), ('B',0.)]),
                   dict([('df',2.), ('C',0.)]),
                   dict([('df',2.), ('C',0.), ('rc',0.)]),
                   dict([('df',2.), ('lambdaH',1.0)]),
                   dict([('df',2.), ('C',0.), ('lambdaH',1.0)]),
                   dict([('df',2.), ('C',0.), ('F',0.), ('lambdaH',1.0)]),
                   dict([('df',2.), ('B',0.), ('C',0.), ('F',0.), ('lambdaH',1.0)])]
    label_list = [r'$d_f=2$, $B=0$',
                  r'$d_f=2$, $C=0$',
                  r'$d_f=2$, $C=0$, $r_c=0$',
                  r'$d_f=2$, $\lambda_h=1$',
                  r'$d_f=2$, $C=0$, $\lambda_h=1$',
                  r'$d_f=2$, $C=0$, $F=0$, $\lambda_h=1$',
                  r'$d_f=2$, $B=0$, $C=0$, $F=0$, $\lambda_h=1$']
    pS_list = []
    pe_list = []

    for fixed_dict in fixed_dicts:
        params, err = fitting.joint_fit([r,Sigma,r,eta], fixed_dict=fixed_dict,
                                        func_type='well-behaved', params0=params0,
                                        verbose=False, show_params=False)
        pS, pe =  separate_params(params, func_type='well-behaved')
        pS_list.append(pS)
        pe_list.append(pe)

    Sigma_errors, eta_errors = get_function_errors(func_type='well-behaved')
    r_errors, Sigma_vals_errors, err = Sigma_errors
    r_errors, eta_vals_errors, err = eta_errors

    labels = [r'$r$',r'$\Sigma(r)$']
    logscale = [False,True]
    Range = [[], []]
    fig, ax = setup_plot(labels, logscale, Range)
    ax.scatter(r,Sigma,label='data',color='k',s=40)
    ax.errorbar(r_errors, Sigma_vals_errors, yerr = err, color='k', ls='none')
    rlist = np.arange(0.5,10,(10.-0.5)/1000)
    for i, label in enumerate(label_list):
        ax.plot(rlist, Sigma_func(rlist, pS_list[i], func_type='well-behaved'),label=label)
    figure_name='Sigma_comparison_wellbehaved.png'
    loc = 'upper right'
    plt.xlim([0.5,9.])
    plt.ylim([0.,10**5.])
    finish_plot(ax, figure_name, loc=loc)

    labels = [r'$r$',r'$\eta(r)$']
    logscale = [False,True]
    Range = [[], []]
    fig, ax = setup_plot(labels, logscale, Range)
    ax.scatter(r,eta,label='data',color='k',s=40)
    ax.errorbar(r_errors, eta_vals_errors, yerr = err, color='k', ls='none')
    rlist = np.arange(0.5,10,(10.-0.5)/1000)
    for i, label in enumerate(label_list):
        ax.plot(rlist, eta_func(rlist, pe_list[i], func_type='well-behaved'),label=label)
    loc = 'lower right'
    figure_name = 'eta_comparison_wellbehaved.png'
    plt.xlim([0.5,9.])
    plt.ylim([10**-2.,10**1.1])
    finish_plot(ax, figure_name, loc=loc)

    return
