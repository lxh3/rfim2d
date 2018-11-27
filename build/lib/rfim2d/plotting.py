import re
import matplotlib.pyplot as plt
import numpy as np

from . import general_use


replace_dict = {"$" : "", "/" : "", "(" : "", ")" : "", "\\" : "", " " : "", "," : ""}


def multiple_replace(adict, text):
    """ 
    Python cookbook function to do multiple replace actions on a string
    """
    # Create a regular expression from all of the dictionary keys
    regex = re.compile("|".join(map(re.escape, adict.keys(  ))))

    # For each match, look up the corresponding value in the dictionary
    return regex.sub(lambda match: adict[match.group(0)], text)


def define_colors(r_list):
    """
    define a random color for each value of r
    """
    num_colors = len(r_list)
    return general_use.generate_colors(num_colors)


def define_figure_name(labels, ext=''):
    """
    define a generic name for the figure to be saved under
    """
    labelx = multiple_replace(replace_dict, labels[0])
    labely = multiple_replace(replace_dict, labels[1])
    return '{}v{}{}'.format(labelx, labely, ext)


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


def finish_plot(ax, figure_name, loc='upper left'):
    """
    Add a legend to the plot, save and close
    """
    # Legend
    legend = ax.legend(loc=loc,  shadow=True)

    # Save figure and close plot 
    plt.savefig(figure_name,bbox_inches='tight')
    plt.close()

    return



def plot_xy(data, labels, logscale=[False,False], Range=[[],[]], colors=None, figure_name=None):
    """
    Plot x vs y for each value of r
    """
    r_list, x_list, y_list = data

    if colors == None:
        colors = define_colors(r_list)

    if figure_name == None:
        figure_name = define_figure_name(labels)

    ax = setup_plot(labels, logscale, Range)

    # For each value of r, plot x vs y with a corresponding color
    for r, x, y, color in zip(r_list, x_list, y_list, colors):
        ax.scatter(x, y, c=color, label=r, lw=0)

    finish_plot(ax, figure_name)

    return


def plot_xy_and_xfofx(data, function, labels, constant, constant_given_r=None, logscale=[False,False], Range=[[],[]], colors=None, figure_name=None):
    """
    Plot x vs y and x vs function(x) for each value of r
    """
    r_list, x_list, y_list = data
    
    if colors == None and r_list != None:
        colors = define_colors(r_list)

    if figure_name == None:
        figure_name = define_figure_name(labels,'_fits')

    ax = setup_plot(labels, logscale, Range)

    if r_list == None:
        ax.scatter(x_list, y_list, color='r', label=labels[1], lw=2)

        minimum_r = np.min(x_list)
        maximum_r = np.max(x_list)
        step_size = (maximum_r-minimum_r)/100.
        r_vals = np.arange(minimum_r, maximum_r, step_size)
        Sigma_vals = function(r_vals, constant)

        ax.plot(r_vals, Sigma_vals, color='k', label=labels[1]+r'$_{scaling}$', lw=1)
        loc='upper right'
    else:
        for r, x, y, cgr, color in zip(r_list, x_list, y_list, constant_given_r, colors):
            ax.scatter(x, y, color=color,label=str(r),lw=0)
            ax.plot(x,function(x, cgr, constant),color=color)
            loc='upper left'

    finish_plot(ax, figure_name, loc)

    return


def plot_collapse(data, function, labels, constant, constant_given_r, logscale=[False,False], Range=[[],[]], colors=None, figure_name=None):
    """
    Plot the scaling collapse of the data (x,y) and fitting function output (x,function(y))  
    """
    r_list, x_list, y_list = data

    if colors == None:
        colors = define_colors(r_list)

    if figure_name == None:
        figure_name = define_figure_name(labels,'_collapse')

    ax = setup_plot(labels, logscale, Range)

    for r, x, y, cgr, color in zip(r_list, x_list, y_list, constant_given_r, colors):
        xscaled, yscaled, yscaled_from_function = function(x, y, cgr, constant)
        ax.scatter(xscaled, yscaled, color=color,label=str(r), lw=0)
        ax.plot(xscaled, yscaled_from_function, color='black', lw=2)

    finish_plot(ax, figure_name)

    return

