from .scaling import Sigma_functional_form, eta_functional_form

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True

plot_sizes = {
    'A': (6, 2.5),
    'dMdh': (9.5, 4),
    'Sigma': (6.5, 4),
    'Sigma_powerlaw': (7.5, 4),
    'Sigma_pitchfork': (10, 4),
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
    return plt


def get_A_text(params, func_type=None):
    """
    Get text associated with the fit of A(s)
    """
    line1 = r'$A(s|r)$ is assumed to take the form:'
    line2 = (r'$A(s|r) = s^{-1}\bigg{(}\frac{s}{\Sigma(r)}\bigg{)}^a '
             r'exp\bigg{(}{-\bigg{(}\frac{s}{\Sigma(r)}\bigg{)}^b}\bigg{)}$')
    a, b = params['a'], params['b']
    line3 = r'where a = {:.4f} and b = {:.4f}'.format(a, b)
    text = '\n'.join([line1, line2, line3])
    return text


def get_dMdh_text(params, func_type=None):
    """
    Get text associated with the fit of dM/dh(h)
    """
    line1 = r'$\frac{d\mathcal{M}}{dh}(h|r)$ is assumed to take the form:'
    line2 = (r'$\frac{d\mathcal{M}}{dh}(h_s) = '
             r'exp\bigg{(}'
             r'\frac{-h_s^2}{(a+bh_s+ch_s^2)^{\frac{d}{2}}}'
             r'\bigg{)}$')
    line3 = r'$h_s = \frac{h-h_{max}(r)}{\eta(r)}$'
    a, b, c, d = params['a'], params['b'], params['c'], params['d']
    line4 = (r'where a = {:.4f}, b = {:.4f}, c = {:.4f}, '
             r'and d = {:.4f}'.format(a, b, c, d))
    text = '\n'.join([line1, line2, line3, line4])
    return text


def get_Sigma_text(params, func_type='wellbehaved'):
    """
    Get text associated with the fit of Sigma(r)
    """
    line1 = 'We have:'
    line2 = Sigma_functional_form(func_type=func_type)
    if func_type == 'powerlaw':
        rS, rc, sS = params['rScale'], params['rc'], params['sScale']
        sigma = params['sigma']
        line3 = (r'where $w=(r-r_c)/r_s$ and '
                 r'$\Sigma$ is given a scale $\Sigma_s$')
        line4 = (r'$r_s$ = {:.4f}, $r_c$ = {:.4f},'
                 r' and $s_s$ = {:.4f}'.format(rS, rc, sS))
        line5 = r'$\sigma$ = {:.4f}'.format(sigma)
        text = '\n'.join([line1, line2, line3, line4, line5])
    else:
        rS, sS = params['rScale'], params['sScale']
        sigmaNu = params['sigmaNu']
        B, F = params['B'], params['F']
        line3 = r'where $w=r/r_s$ and $\Sigma$ is given a scale $\Sigma_s$'
        line4 = r'$r_s$ = {:.4f} and $s_s$ = {:.4f}'.format(rS, sS)
        line5 = r'$\sigma\nu$ = {:.4f}'.format(sigmaNu)
        line6 = 'B = {:.4f} and F = {:.4f}'.format(B, F)
        text = '\n'.join([line1, line2, line3, line4, line5, line6])
    return text


def get_eta_text(params, func_type='wellbehaved'):
    """
    Get text associated with the fit of eta(r)
    """
    line1 = 'We have:'
    line2 = eta_functional_form(func_type=func_type)
    if func_type == 'powerlaw':
        rS, rc, etaS = params['rScale'], params['rc'], params['etaScale']
        betaDelta = params['betaDelta']
        line3 = (r'where $w=(r-r_c)/r_s$ and '
                 r'$\eta$ is given a scale $\eta_s$')
        line4 = (r'$r_s$ = {:.4f}, $r_c$ = {:.4f}, '
                 r'and $\eta_s$ = {:.4f}'.format(rS, rc, etaS))
        line5 = r'$\beta\delta$ = {:.4f}'.format(betaDelta)
        text = '\n'.join([line1, line2, line3, line4, line5])
    else:
        rS, etaS = params['rScale'], params['etaScale']
        betaDeltaOverNu = params['betaDeltaOverNu']
        B, C = params['B'], params['C']
        line3 = r'where $w=r/r_s$ and $\eta$ is given a scale $\eta_s$'
        line4 = (r'$r_s$ = {:.4f} and '
                 r'$\eta_s$ = {:.4f}'.format(rS, etaS))
        line5 = r'$\beta\delta/\nu$ = {:.4f}'.format(betaDeltaOverNu)
        line6 = 'B = {:.4f} and C = {:.4f}'.format(B, C)
        text = '\n'.join([line1, line2, line3, line4, line5, line6])
    return text


def get_joint_text(params, func_type='wellbehaved'):
    """
    Get text associated with the joint fit of Sigma(r) and eta(r)
    """
    line1 = 'We have:'
    line2 = Sigma_functional_form(func_type=func_type)
    line3 = eta_functional_form(func_type=func_type)

    if func_type == 'powerlaw':
        rS, rc, sS, etaS = [params['rScale'], params['rc'],
                            params['sScale'], params['etaScale']]
        sigma, betaDelta = params['sigma'], params['betaDelta']
        line4 = (r'where $w=(r-r_c)/r_s$ and $\eta$ and $\Sigma$ '
                 r'are given a scale $\eta_s$ and $\Sigma_s$ respectively')
        line5 = (r'$r_s$ = {:.4f}, $r_c$ = {:.4f},  $s_s$ = {:.4f}, '
                 r'$\eta_s$ = {:.4f}'.format(rS, rc, sS, etaS))
        line6 = (r'$\sigma$ = {:.4f}, '
                 r'$\beta\delta$ = {:.4f}'.format(sigma, betaDelta))
        text = '\n'.join([line1, line2, line3, line4, line5, line6])
    else:
        rS, sS, etaS = [params['rScale'], params['sScale'],
                        params['etaScale']]
        sigmaNu, betaDeltaOverNu = [params['sigmaNu'],
                                    params['betaDeltaOverNu']]
        B, C, F = params['B'], params['C'], params['F']
        line4 = (r'where $w=r/r_s$ and $\eta$ and $\Sigma$ '
                 r'are given a scale $\eta_s$ and $\Sigma_s$ respectively')
        line5 = (r'$r_s$ = {:.4f}, $s_s$ = {:.4f}, '
                 r'$\eta_s$ = {:.4f}'.format(rS, sS, etaS))
        line6 = (r'$\sigma\nu$ = {:.4f}, $\beta\delta/\nu$ = {:.4f}'
                 r''.format(sigmaNu, betaDeltaOverNu))
        line7 = 'B = {:.4f}, C = {:.4f}, and F = {:.4f}'.format(B, C, F)
        text = '\n'.join([line1, line2, line3, line4, line5, line6, line7])
    return text


def get_text(params, fit_type, func_type=None):
    func = {
        'A': get_A_text,
        'dMdh': get_dMdh_text,
        'Sigma': get_Sigma_text,
        'eta': get_eta_text,
        'joint': get_joint_text
        }
    try:
        function = func.get(fit_type)
        return function(params, func_type=func_type)
    except KeyError:
        text = ('Functional form requested not recognized. '
                'Printing parameter dict: '+str(params))
        return text


def print_fit_info(params, fit_type, func_type='wellbehaved',
                   filename=None, show=True):

    text = get_text(params, fit_type, func_type=func_type)

    ft = func_type
    if fit_type != 'joint' and (ft == 'powerlaw' or ft == 'pitchfork'):
        fit_type = fit_type+'_'+func_type

    plt = make_plot(text, plot_sizes[fit_type])

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.close()

    return
