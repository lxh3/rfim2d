import numpy as np

from scipy.special import gamma, gammaincc


def As_Scaling(s, constant_with_r, constant):
    """
    Assumed form of the area weighted size distribution (A)
    times the avalanche size (s) as a function of the avalanche size,
    Sigma(r), and unknown constants a and b
    """
    Sigma = constant_with_r
    a, b = constant

    sscaled = s/Sigma
    As = sscaled**a*np.exp(-sscaled**b)
    AsNorm = gamma(a/b)*gammaincc(a/b, Sigma**(-2*b))/b
    return As/AsNorm


def As_Collapse(s, As, constant_with_r, constant):
    """
    Function to obtain the values to plot in order to collapse A
    """
    xscaled = s/constant_with_r
    yscaled = As
    yscaled_from_function = As_Scaling(s, constant_with_r, constant)
    return xscaled, yscaled, yscaled_from_function


def dMdh_Scaling(h, constant_with_r, constant):
    """
    Assumed form of the derivative of the magnetization (M) with respect
    to the field (h) as a function of the field and unknown constants
    a, b, c and power
    """
    hmax, eta = constant_with_r
    a, b, c, d = constant

    hscaled = (h-hmax)/eta
    dMdh = np.exp(-hscaled**2/(a+b*hscaled+c*hscaled**2)**(d/2.))
    dh = 0.01
    dMdhNorm = dh*sum(dMdh)/2.
    return dMdh/dMdhNorm


def dMdh_Collapse(h, dMdh, constant_with_r, constant):
    """
    Function to obtain the values to plot in order to collapse dM/dh
    """
    hmax, eta = constant_with_r
    xscaled = (h-hmax)/eta
    yscaled = eta*dMdh
    yscaled_from_function = eta*dMdh_Scaling(h, constant_with_r, constant)
    return xscaled, yscaled, yscaled_from_function


def Sigma_functional_form(func_type='well-behaved'):
    """
    Get line with the correct functional form of Sigma(w)
    """
    if func_type == 'power law':
        form = r'$\Sigma(w)=w^{-\frac{1}{\sigma}}$'
    elif func_type == 'truncated':
        form = (r'$\Sigma(w)'
                r'=\big{(}\frac{1}{w}+B\big{)}^{-B d_f+C}'
                r'exp\big{(}\frac{d_f}{w}\big{)}$')
    elif func_type == 'well-behaved':
        form = (r'$\Sigma(w)=w^{B d_f-C}'
                r'exp\big{(}\frac{d_f}{w}+B C w\big{)}$')
    elif func_type == 'pitchfork':
        form = (r'$\Sigma(w)=w^{B d_f}'
                r'(1+Bw^2)^{-\frac{B d_f}{2}}'
                r'exp\big{(}\frac{d_f}{2w^2}+\frac{C}{w}'
                r'+\sqrt{B}CArcTan[\sqrt{B}w]\big{)}$')
    else:
        print('functional form requested not recognized')
        form = 'unknown functional form'
    return form


def get_Sigma_powerlaw_params(param_dict):
    """
    Extract and return parameters from dictionary for powerlaw form
    """
    rScale = param_dict['rScale']
    rc = param_dict['rc']
    sScale = param_dict['sScale']
    sigma = param_dict['sigma']
    return rScale, rc, sScale, sigma


def get_Sigma_params(param_dict):
    """
    Extract and return parameters from dictionary for all 
    forms other than the powerlaw form
    """
    rScale = param_dict['rScale']
    rc = param_dict['rc']
    sScale = param_dict['sScale']
    df = param_dict['df']
    B = param_dict['B']
    C = param_dict['C']
    return rScale, rc, sScale, df, B, C


def Sigma_func(r_list, param_dict, func_type='well-behaved', scaled=False):
    """
    Nonlinear scaling variable function determined from
    the flow equations of the disorder, w, and avalanche size, s
    Input:
        r_list - list of r values
        param_dict - dictionary of parameters on which Sigma(r) is dependent
        func_type - which form of Sigma(r) to use. Options:
            'power law' - Sigma(r) derived with dw/dl = (1/nu) w
            'truncated' - Sigma(r) derived with dw/dl = w^2 + B w^3
            'well-behaved' - Sigma(r) derived with dw/dl = w^2 / (1 + B w)
            'pitchfork' - Sigma(r) derived with dw/dl = w^3 + B w^5
        scaled - flag for whether to use the Sigma derived with the
                 'truncated' dw/dl scaled for comparison with the
                 'well-behaved' form at small r
    Output:
        Sigma(r) values for the given inputs
    """
    if func_type == 'power law':
        rScale, rc, sScale, sigma = get_Sigma_powerlaw_params(param_dict)
        r = np.array(r_list)
        w = (r-rc)/rScale
        return abs(sScale)*w**(-1/sigma)
    else:
        rScale, rc, sScale, df, B, C = get_Sigma_params(param_dict)
        r = np.array(r_list)
        w = (r-rc)/rScale
        if func_type == 'truncated':
            Sigma = (abs(sScale)
                     * np.exp(df/w) * (B+(1/w)) ** (C-B*df))
            if scaled:
                Sigma = np.exp(B**2*w*df)*Sigma
            return Sigma
        elif func_type == 'well-behaved':
            Sigma = (abs(sScale)
                     * np.exp(B*C*w + df/w)
                     * w ** (B*df-C))
            if scaled:
                Sigma = np.exp(-B**2*w*df)*Sigma
            return Sigma
        elif func_type == 'pitchfork':
            exponential_piece = (np.exp((C/w)+(df/(2*w**2))
                                 + np.sqrt(B) * C * np.arctan(np.sqrt(B)*w)))
            powerlaw_piece = w ** (B*df) * (1+B*w**2) ** (-B*df/2)
            return abs(sScale)*exponential_piece*powerlaw_piece
        else:
            print("functional type not recognized,",
                  " using 'well-behaved' form")
            Sigma = (abs(sScale) * np.exp(B*C*w + df/(w))
                     * w ** ((B*df)-C))
            return Sigma


def eta_functional_form(func_type='well-behaved'):
    """
    Get line with the correct functional form of Sigma(w)
    """
    if func_type == 'power law':
        form = r'$\eta(w)=w^{\beta\delta}$'
    elif func_type == 'truncated':
        form = (r'$\eta(w)='
                r'\big{(}\frac{1}{w}+B\big{)}^{B \lambda_h-F}'
                r'exp\big{(}-\frac{\lambda_h}{w}\big{)}$')
    elif func_type == 'well-behaved':
        form = (r'$\eta(w) = w^{-B \lambda_h+F}'
                r'exp\big{(}-\frac{\lambda_h}{w}'
                r'-BFw\big{)}$')
    elif func_type == 'pitchfork':
        form = (r'$\eta(w)=w^{-B\lambda_h}'
                r'(1+Bw^2)^{\frac{B\lambda_h}{2}}'
                r'exp\big{(}-\frac{\lambda_h}{2w^2}'
                r'-\frac{F}{w}-\sqrt{B}FArcTan[\sqrt{B}w]\big{)}$')
    else:
        print('functional form requested not recognized')
        form = 'unknown functional form'
    return form


def get_eta_powerlaw_params(param_dict):
    """
    Extract and return parameters from dictionary for powerlaw form
    """
    rScale = param_dict['rScale']
    rc = param_dict['rc']
    etaScale = param_dict['etaScale']
    betaDelta = param_dict['betaDelta']
    return rScale, rc, etaScale, betaDelta


def get_eta_params(param_dict):
    """
    Extract and return parameters from dictionary for all 
    forms other than the powerlaw form
    """
    rScale = param_dict['rScale']
    rc = param_dict['rc']
    etaScale = param_dict['etaScale']
    lambdaH = param_dict['lambdaH']
    B = param_dict['B']
    F = param_dict['F']
    return rScale, rc, etaScale, lambdaH, B, F


def eta_func(r_list, param_dict, func_type='well-behaved', scaled=False):
    """
    Nonlinear scaling variable function determined from
    the flow equations of the disorder, w, and field, h
    Input:
        r_list - list of r values
        param_dict - dictionary of parameters on which eta(r) is dependent
        func_type - which form of eta(r) to use. Options:
            'power law' - eta(r) derived with dw/dl = (1/nu) w
            'truncated' - eta(r) derived with dw/dl = w^2 + B w^3
            'well-behaved' - eta(r) derived with dw/dl = w^2 / (1 + B w)
            'pitchfork' - eta(r) derived with dw/dl = w^3 + B w^5
        scaled - flag for whether to use the eta derived with the
                 'truncated' dw/dl scaled for comparison with the
                 'well-behaved' form at small r
    Output:
        eta(r) values for the given inputs
    """
    if func_type == 'power law':
        rScale, rc, etaScale, betaDelta = get_eta_powerlaw_params(param_dict)
        r = np.array(r_list)
        w = (r-rc)/rScale
        return etaScale*w**(betaDelta)
    else:
        rScale, rc, etaScale, lambdaH, B, F = get_eta_params(param_dict)
        r = np.array(r_list)
        w = (r-rc)/rScale
        if func_type == 'truncated':
            eta = (etaScale*np.exp(-lambdaH/w)
                   * (B+(1/w)) ** (-F+B*lambdaH))
            if scaled:
                eta = np.exp(-B**2*w*lambdaH)*eta
            return eta
        elif func_type == 'well-behaved':
            eta = (etaScale*np.exp(-B*F*w-lambdaH/w)
                   * w ** (-B*lambdaH+F))
            if scaled:
                eta = np.exp(B**2*w*lambdaH)*eta
            return eta
        elif func_type == 'pitchfork':
            exponential_piece = (np.exp(-(F/w)-(lambdaH/(2*w**2))
                                 - np.sqrt(B) * F * np.arctan(np.sqrt(B)*w)))
            powerlaw_piece = (w**(-B*lambdaH)
                              * (1+B*w**2) ** (B*lambdaH/2))
            return etaScale*exponential_piece*powerlaw_piece
        else:
            print("functional type not recognized,",
                  " using 'well-behaved' form")
            eta = (etaScale*np.exp(-B*F*w-lambdaH/w)
                   * w ** (-B*lambdaH+F))
            return eta
