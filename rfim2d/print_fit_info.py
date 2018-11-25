def print_A_fit_info(num_curves, params):
    """
    print functions and parameters associated with the fit of A(s)
    """
    try:
        Sigma = params[:num_curves]
        a, b = params[num_curves:]
        print(r'$A(s|r)$ is assumed to take the form:')
        print(r'$A(s|r) = s^{-1}\big{(}\frac{s}{\Sigma(r)}\big{)}^a exp\big{(}{-\big{(}\frac{s}{\Sigma(r)}\big{)}^b}\big{)}$')
        print('where a = {} and b = {}'.format(a, b))
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
        print(r'$\frac{d\mathcal{M}}{dh}(h|r)$ is assumed to take the form:')
        print(r'$\frac{d\mathcal{M}}{dh}(h_s) = exp\bigg{(}\frac{-h_s^2}{(a+bh_s+ch_s^2)^{\frac{d}{2}}}\bigg{)}$')
        print(r'$h_s = \frac{h-h_{max}(r)}{\eta(r)}$')
        print('where a = {}, b = {}, c = {}, and d = {}'.format(a, b, c, d))
    except ValueError:
        print('num_curves and params are of incompatible lengths') 


def print_joint_fit_info(params, sigmaNu_fixed):
    """
    print functions and parameters associated with the joint fit of Sigma(r) and eta(r)
    """
    try:
        if sigmaNu_fixed:
            sigmaNu = 0.5
            rScale, sScale, etaScale, betaDeltaOverNu, B, C, F = params
        else:
            rScale, sScale, etaScale, sigmaNu, betaDeltaOverNu, B, C, F = params
        print('We have:')
        print(r'$\Sigma(w)=w^{-\frac{B}{\sigma\nu}-F}exp\big{(}\frac{1}{w\sigma\nu}-B F w\big{)}$')
        print(r'$\eta(w) = w^{C+B\frac{\beta\delta}{\nu}}exp\big{(}-\frac{\beta\delta}{\nu}\frac{1}{w}+BCw\big{)}$')
        print(r'where $w=r/r_{scale}$ and $\eta$ and $s$ are given a scale $\eta_{scale}$ and $s_{scale}$ respectively')
        print('rScale = {}, sScale = {}, etaScale = {}'.format(rScale, sScale, etaScale))
        print(r'$\sigma\nu$ = {}, $\beta\delta/\nu$ = {}'.format(sigmaNu, betaDeltaOverNu))
        print('B = {}, C = {}, and F = {}'.format(B,C,F))
    except ValueError:
        print('incorrect number of parameters supplied') 
