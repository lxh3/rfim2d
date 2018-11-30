import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True
from .general_use import add_sigmaNu

def print_A_fit_info(num_curves, params):
    """
    print functions and parameters associated with the fit of A(s)
    """
    try:
        Sigma = params[:num_curves]
        a, b = params[num_curves:]
        line1 = r'$A(s|r)$ is assumed to take the form:'
        line2 = r'$A(s|r) = s^{-1}\bigg{(}\frac{s}{\Sigma(r)}\bigg{)}^a exp\bigg{(}{-\bigg{(}\frac{s}{\Sigma(r)}\bigg{)}^b}\bigg{)}$'
        line3 = r'where a = {:.4f} and b = {:.4f}'.format(a, b)
        plt.figure(figsize=(7, 2))
        plt.text(0.1, 0.25, '\n'.join([line1,line2,line3]), fontsize=14)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.savefig('A_fit_info.png')
    except ValueError:
        print('num_curves and params are of incompatible lengths')  


def print_dMdh_fit_info(num_curves, params):
    """
    print functions and parameters associated with the fit of dM/dh(h)
    """
    try:
        hMax = params[:num_curves]
        eta = params[num_curves:2*num_curves]
        a, b, c, d = params[2*num_curves:]
        line1 = r'$\frac{d\mathcal{M}}{dh}(h|r)$ is assumed to take the form:'
        line2 = r'$\frac{d\mathcal{M}}{dh}(h_s) = exp\bigg{(}\frac{-h_s^2}{(a+bh_s+ch_s^2)^{\frac{d}{2}}}\bigg{)}$'
        line3 = r'$h_s = \frac{h-h_{max}(r)}{\eta(r)}$'
        line4 = r'where a = {:.4f}, b = {:.4f}, c = {:.4f}, and d = {:.4f}'.format(a, b, c, d)
        plt.figure(figsize=(7.25, 2.75))
        plt.text(0.1, 0.25, '\n'.join([line1,line2,line3,line4]), fontsize=14)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.savefig('dMdh_fit_info.png')
    except ValueError:
        print('num_curves and params are of incompatible lengths') 

def print_Sigma_fit_info(params, sigmaNu_fixed=True, simple=False):
    """
    print functions and parameters associated with the fit of Sigma(r)
    """
    try:
        if sigmaNu_fixed:
            params = add_sigmaNu(params)
        rScale, sScale, sigmaNu, B, F = params
        line1 = 'We have:'
        if simple:
            line2 = r'$\Sigma(w)=\big{(}\frac{1}{w}+B\big{)}^{-B\frac{1}{\sigma\nu}+F}exp\big{(}\frac{1}{w\sigma\nu}\big{)}$'
        else:
            line2 = r'$\Sigma(w)=w^{\frac{B}{\sigma\nu}-F}exp\big{(}\frac{1}{w\sigma\nu}+B F w\big{)}$'
        line3 = r'where $w=r/r_s$ and $s$ is given a scale $s_s$'
        line4 = r'$r_s$ = {:.4f} and $s_s$ = {:.4f}'.format(rScale, sScale)
        line5 = r'$\sigma\nu$ = {:.4f}'.format(sigmaNu)
        line6 = 'B = {:.4f} and F = {:.4f}'.format(B,F)
        plt.figure(figsize=(6.5, 3.5))
        plt.text(0.1, 0.25, '\n'.join([line1,line2,line3,line4,line5,line6]), fontsize=14)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.savefig('Sigma_fit_info.png')
    except ValueError:
        print('incorrect number of parameters supplied') 

def print_eta_fit_info(params, simple=False):
    """
    print functions and parameters associated with the fit of eta(r)
    """
    try:
        rScale, etaScale, betaDeltaOverNu, B, C = params
        line1 = 'We have:'
        if simple:
            line2 = r'$\eta(w)=\big{(}\frac{1}{w}+B\big{)}^{B\frac{\beta\delta}{\nu}-C}exp\big{(}-\frac{\beta\delta}{w\nu}\big{)}$'
        else:
            line2 = r'$\eta(w) = w^{-B\frac{\beta\delta}{\nu}+C}exp\big{(}-\frac{\beta\delta}{\nu}\frac{1}{w}-BCw\big{)}$'
        line3 = r'where $w=r/r_s$ and $\eta$ is given a scale $\eta_s$'
        line4 = r'$r_s$ = {:.4f} and $\eta_s$ = {:.4f}'.format(rScale, etaScale)
        line5 = r'$\beta\delta/\nu$ = {:.4f}'.format(betaDeltaOverNu)
        line6 = 'B = {:.4f} and C = {:.4f}'.format(B,C)
        plt.figure(figsize=(6.5, 3.5))
        plt.text(0.1, 0.25, '\n'.join([line1,line2,line3,line4,line5,line6]), fontsize=14)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.savefig('eta_fit_info.png')
    except ValueError:
        print('incorrect number of parameters supplied') 
        
def print_joint_fit_info(params, sigmaNu_fixed=True, simple=False):
    """
    print functions and parameters associated with the joint fit of Sigma(r) and eta(r)
    """
    try:
        if sigmaNu_fixed:
            params = add_sigmaNu(params)
        rScale, sScale, etaScale, sigmaNu, betaDeltaOverNu, B, C, F = params
        line1 = 'We have:'
        if simple:
            line2 = r'$\Sigma(w)=\big{(}\frac{1}{w}+B\big{)}^{-B\frac{1}{\sigma\nu}+F}exp\big{(}\frac{1}{w\sigma\nu}\big{)}$'
            line3 = r'$\eta(w)=\big{(}\frac{1}{w}+B\big{)}^{B\frac{\beta\delta}{\nu}-C}exp\big{(}-\frac{\beta\delta}{w\nu}\big{)}$'
        else:
            line2 = r'$\Sigma(w)=w^{\frac{B}{\sigma\nu}-F}exp\big{(}\frac{1}{w\sigma\nu}+B F w\big{)}$'
            line3 = r'$\eta(w) = w^{-B\frac{\beta\delta}{\nu}+C}exp\big{(}-\frac{\beta\delta}{\nu}\frac{1}{w}-BCw\big{)}$'
        line4 = r'where $w=r/r_s$ and $\eta$ and $s$ are given a scale $\eta_s$ and $s_s$ respectively'
        line5 = r'$r_s$ = {:.4f}, $s_s$ = {:.4f}, $\eta_s$ = {:.4f}'.format(rScale, sScale, etaScale)
        line6 = r'$\sigma\nu$ = {:.4f}, $\beta\delta/\nu$ = {:.4f}'.format(sigmaNu, betaDeltaOverNu)
        line7 = 'B = {:.4f}, C = {:.4f}, and F = {:.4f}'.format(B,C,F)
        plt.figure(figsize=(9, 3.5))
        plt.text(0.1, 0.25, '\n'.join([line1,line2,line3,line4,line5,line6,line7]), fontsize=14)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.savefig('joint_fit_info.png')
    except ValueError:
        print('incorrect number of parameters supplied') 
