import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True

from .scaling import Sigma_functional_form, eta_functional_form

plot_sizes = {
    'A': (6, 2.5),
    'dMdh': (9.5, 4),
    'Sigma': (6.5, 4),
    'Sigma_powerlaw': (7.5, 4),
    'Sigma_pitchfork':(10, 4),
    'eta': (6.5, 4),
    'eta_powerlaw': (7.5, 4),
    'eta_pitchfork': (11, 4),
    'joint': (11, 5)
}

def make_plot(text, figsize, text_loc=[0.1, 0.25], fontsize=18):
    """
    Plot text associated with the requested fit
    """
    plt.figure(figsize=figsize)
    plt.text(text_loc[0], text_loc[1], text, fontsize=fontsize)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return ax

def get_A_text(params):
    """
    Get text associated with the fit of A(s)
    """
    line1 = r'$A(s|r)$ is assumed to take the form:'
    line2 = r'$A(s|r) = s^{-1}\bigg{(}\frac{s}{\Sigma(r)}\bigg{)}^a exp\bigg{(}{-\bigg{(}\frac{s}{\Sigma(r)}\bigg{)}^b}\bigg{)}$'
    line3 = r'where a = {:.4f} and b = {:.4f}'.format(params['a'], params['b'])
    text = '\n'.join([line1,line2,line3])
    return text

def get_dMdh_text(params):
    """
    Get text associated with the fit of dM/dh(h)
    """
    line1 = r'$\frac{d\mathcal{M}}{dh}(h|r)$ is assumed to take the form:'
    line2 = r'$\frac{d\mathcal{M}}{dh}(h_s) = exp\bigg{(}\frac{-h_s^2}{(a+bh_s+ch_s^2)^{\frac{d}{2}}}\bigg{)}$'
    line3 = r'$h_s = \frac{h-h_{max}(r)}{\eta(r)}$'
    line4 = r'where a = {:.4f}, b = {:.4f}, c = {:.4f}, and d = {:.4f}'.format(params['a'], params['b'], params['c'], params['d'])
    text = '\n'.join([line1,line2,line3,line4])
    return text

def get_Sigma_text(params, func_type='wellbehaved'):
    """
    Get text associated with the fit of Sigma(r)
    """
    line1 = 'We have:'
    line2 = Sigma_functional_form(func_type=func_type)
    if func_type == 'powerlaw':
        line3 = r'where $w=(r-r_c)/r_s$ and $\Sigma$ is given a scale $\Sigma_s$'
        line4 = r'$r_s$ = {:.4f}, $r_c$ = {:.4f}, and $s_s$ = {:.4f}'.format(params['rScale'], params['rc'], params['sScale'])
        line5 = r'$\sigma$ = {:.4f}'.format(params['sigma'])
        text = '\n'.join([line1,line2,line3,line4,line5])
    else:
        line3 = r'where $w=r/r_s$ and $\Sigma$ is given a scale $\Sigma_s$'
        line4 = r'$r_s$ = {:.4f} and $s_s$ = {:.4f}'.format(params['rScale'], params['sScale'])
        line5 = r'$\sigma\nu$ = {:.4f}'.format(params['sigmaNu'])
        line6 = 'B = {:.4f} and F = {:.4f}'.format(params['B'],params['F'])
        text = '\n'.join([line1,line2,line3,line4,line5,line6])
    return text

def get_eta_text(params, func_type='wellbehaved'):
    """
    Get text associated with the fit of eta(r)
    """
    line1 = 'We have:'
    line2 = eta_functional_form(func_type=func_type)
    if func_type == 'powerlaw':
        line3 = r'where $w=(r-r_c)/r_s$ and $\eta$ is given a scale $\eta_s$'
        line4 = r'$r_s$ = {:.4f}, $r_c$ = {:.4f}, and $\eta_s$ = {:.4f}'.format(params['rScale'], params['rc'], params['etaScale'])
        line5 = r'$\beta\delta$ = {:.4f}'.format(params['betaDelta'])
        text = '\n'.join([line1,line2,line3,line4,line5])
    else:
        line3 = r'where $w=r/r_s$ and $\eta$ is given a scale $\eta_s$'
        line4 = r'$r_s$ = {:.4f} and $\eta_s$ = {:.4f}'.format(params['rScale'], params['etaScale'])
        line5 = r'$\beta\delta/\nu$ = {:.4f}'.format(params['betaDeltaOverNu'])
        line6 = 'B = {:.4f} and C = {:.4f}'.format(params['B'], params['C'])
        text = '\n'.join([line1,line2,line3,line4,line5,line6])
    return text

def get_joint_text(params, func_type='wellbehaved'):
    """
    Get text associated with the joint fit of Sigma(r) and eta(r)
    """
    line1 = 'We have:'
    line2 = Sigma_functional_form(func_type=func_type)
    line3 = eta_functional_form(func_type=func_type)

    if func_type == 'powerlaw':
        line4 = r'where $w=(r-r_c)/r_s$ and $\eta$ and $\Sigma$ are given a scale $\eta_s$ and $\Sigma_s$ respectively'
        line5 = r'$r_s$ = {:.4f}, $r_c$ = {:.4f},  $s_s$ = {:.4f}, $\eta_s$ = {:.4f}'.format(params['rScale'], params['rc'], params['sScale'], params['etaScale'])
        line6 = r'$\sigma$ = {:.4f}, $\beta\delta$ = {:.4f}'.format(params['sigma'], params['betaDelta'])
        text = '\n'.join([line1,line2,line3,line4,line5,line6])
    else:
        line4 = r'where $w=r/r_s$ and $\eta$ and $\Sigma$ are given a scale $\eta_s$ and $\Sigma_s$ respectively'
        line5 = r'$r_s$ = {:.4f}, $s_s$ = {:.4f}, $\eta_s$ = {:.4f}'.format(params['rScale'], params['sScale'], params['etaScale'])
        line6 = r'$\sigma\nu$ = {:.4f}, $\beta\delta/\nu$ = {:.4f}'.format(params['sigmaNu'], params['betaDeltaOverNu'])
        line7 = 'B = {:.4f}, C = {:.4f}, and F = {:.4f}'.format(params['B'], params['C'], params['F'])
        text = '\n'.join([line1,line2,line3,line4,line5,line6,line7])
    return text


def print_fit_info(params, fit_type, func_type='wellbehaved', filename=None, show=True):

    if fit_type=='A':
        text = get_A_text(params)

    elif fit_type=='dMdh':
        text = get_dMdh_text(params)

    elif fit_type=='Sigma':
        text = get_Sigma_text(params, func_type=func_type)

    elif fit_type=='eta':
        text = get_eta_text(params, func_type=func_type)

    elif fit_type=='joint':
        text = get_joint_text(params, func_type=func_type)

    else:
        text = 'Functional form requested not recognized. Printing parameter dict: '+str(params)
             
    if fit_type!='joint' and (func_type=='powerlaw' or func_type=='pitchfork'):
        fit_type = fit_type+'_'+func_type
    
    ax = make_plot(text, plot_sizes[fit_type])

    if filename != None:
        plt.savefig(filename)
    if show:
        plt.show()
    
    plt.close()

    return
