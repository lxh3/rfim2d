import numpy as np

from scipy.special import gamma, gammaincc

def As_Scaling(s, constant_with_r, constant):
    """
    Assumed form of the area weighted size distribution (A) times the avalanche size (s) 
    as a function of the avalanche size, Sigma(r), and unknown constants a and b
    """
    Sigma = constant_with_r
    a, b = constant

    sscaled = s/Sigma 
    As = sscaled**a*np.exp(-sscaled**b) 
    AsNorm = gamma(a/b)*gammaincc(a/b,Sigma**(-2*b))/b
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
    Assumed form of the derivative of the magnetization (M) with respect to the field (h)
    as a function of the field and unknown constants a, b, c and power
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

def Sigma_functional_form(func_type='wellbehaved'):
    """
    Get line with the correct functional form of Sigma(w)
    """
    if func_type=='powerlaw':
        form = r'$\Sigma(w)=w^{-\frac{1}{\sigma}}$'
    elif func_type=='simple':
        form = r'$\Sigma(w)=\big{(}\frac{1}{w}+B\big{)}^{-B\frac{1}{\sigma\nu}+F}exp\big{(}\frac{1}{w\sigma\nu}\big{)}$'
    elif func_type=='wellbehaved':
        form = r'$\Sigma(w)=w^{\frac{B}{\sigma\nu}-F}exp\big{(}\frac{1}{w\sigma\nu}+B F w\big{)}$'
    elif func_type=='pitchfork':
        form =  r'$\Sigma(w)=w^{\frac{B}{\sigma\nu}}(1+Bw^2)^{-\frac{B}{2\sigma\nu}}exp\big{(}\frac{1}{2w^2\sigma\nu}+\frac{F}{w}+\sqrt{B}FArcTan[\sqrt{B}w]\big{)}$'
    else:
        print('functional form requested not recognized')
        form = 'unknown functional form'
    return form

def Sigma_func(r_list, constant, func_type='wellbehaved', scaled=False):
    """
    Nonlinear scaling variable function determined from 
    the flow equations of the disorder, w, and avalanche size, s
    Input:
        r_list - list of r values
        constant - constants on which Sigma(r) is dependent
        func_type - which form of Sigma(r) to use. Options:
            'powerlaw' - Sigma(r) derived with dw/dl = (1/nu) w
            'simple' - Sigma(r) derived with dw/dl = w^2 + B w^3
            'wellbehaved' - Sigma(r) derived with dw/dl = w^2 / (1 + B w)
            'pitchfork' - Sigma(r) derived with dw/dl = w^3 + B w^5
        scaled - flag for whether to use the Sigma derived with the 'simple' dw/dl
                 scaled for comparison with the 'wellbehaved' form at small r
    Output:
        Sigma(r) values for the given inputs
    """
    if func_type=='powerlaw':
        rScale, sScale, sigma, rc = constant
        r = np.array(r_list)
        w = (r-rc)/rScale
        return abs(sScale)*w**(-1/sigma)
    else:
        rScale, sScale, sigmaNu, B, F = constant
        r = np.array(r_list)
        w = r/rScale
        if func_type=='simple':
            Sigma = abs(sScale)*np.exp(1/(w*sigmaNu))*(B+(1/w))**(F-(B/sigmaNu))
            if scaled:
                Sigma = np.exp((B**2*w)/sigmaNu)*Sigma
            return Sigma 
        elif func_type=='wellbehaved':
            Sigma = abs(sScale)*np.exp(B*F*w + 1/(w*sigmaNu))*w**((B/sigmaNu)-F)
            if scaled:
                Sigma = np.exp((-B**2*w)/sigmaNu)*Sigma
            return Sigma
        elif func_type=='pitchfork':
            exponential_piece = np.exp((F/w)+(1/(2*sigmaNu*w**2))+np.sqrt(B)*F*np.arctan(np.sqrt(B)*w))
            powerlaw_piece = w**(B/sigmaNu)*(1+B*w**2)**(-B/(2*sigmaNu))
            return abs(sScale)*exponential_piece*powerlaw_piece
        else:
            print("functional type not recognized, using 'wellbehaved' form")
            return abs(sScale)*np.exp(B*F*w + 1/(w*sigmaNu))*w**((B/sigmaNu)-F)

def eta_functional_form(func_type='wellbehaved'):
    """
    Get line with the correct functional form of Sigma(w)
    """
    if func_type=='powerlaw':
        form = r'$\eta(w)=w^{\beta\delta}$'
    elif func_type=='simple':
        form = r'$\eta(w)=\big{(}\frac{1}{w}+B\big{)}^{B\frac{\beta\delta}{\nu}-C}exp\big{(}-\frac{\beta\delta}{w\nu}\big{)}$'
    elif func_type=='wellbehaved':
        form = r'$\eta(w) = w^{-B\frac{\beta\delta}{\nu}+C}exp\big{(}-\frac{\beta\delta}{\nu}\frac{1}{w}-BCw\big{)}$'
    elif func_type=='pitchfork':
        form =  r'$\eta(w)=w^{-B\frac{\beta\delta}{\nu}}(1+Bw^2)^{\frac{B\beta\delta}{2\nu}}exp\big{(}-\frac{\beta\delta}{2w^2\nu}-\frac{C}{w}-\sqrt{B}CArcTan[\sqrt{B}w]\big{)}$'
    else:
        print('functional form requested not recognized')
        form = 'unknown functional form'
    return form

def eta_func(r_list, constant, func_type='wellbehaved', scaled=False):
    """
    Nonlinear scaling variable function determined from 
    the flow equations of the disorder, w, and field, h
    Input:
        r_list - list of r values
        constant - constants on which eta(r) is dependent
        func_type - which form of eta(r) to use. Options:
            'powerlaw' - eta(r) derived with dw/dl = (1/nu) w
            'simple' - eta(r) derived with dw/dl = w^2 + B w^3
            'wellbehaved' - eta(r) derived with dw/dl = w^2 / (1 + B w)
            'pitchfork' - eta(r) derived with dw/dl = w^3 + B w^5
        scaled - flag for whether to use the eta derived with the 'simple' dw/dl
                 scaled for comparison with the 'wellbehaved' form at small r
    Output:
        eta(r) values for the given inputs
    """
    if func_type=='powerlaw':
        rScale, etaScale, betaDelta, rc = constant
        r = np.array(r_list)
        w = (r-rc)/rScale
        return etaScale*w**(betaDelta)
    else:
        rScale, etaScale, betaDeltaOverNu, B, C = constant
        r = np.array(r_list)
        w = r/rScale
        if func_type=='simple':
            eta = etaScale*np.exp(-betaDeltaOverNu/w)*(B+(1/w))**(-C+B*betaDeltaOverNu)
            if scaled:
                eta = np.exp(-B**2*w*betaDeltaOverNu)*eta
            return eta
        elif func_type=='wellbehaved':
            eta = etaScale*np.exp(-B*C*w-betaDeltaOverNu/w)*w**(-B*betaDeltaOverNu+C)
            if scaled:
                eta = np.exp(B**2*w*betaDeltaOverNu)*eta
            return eta
        elif func_type=='pitchfork':
            exponential_piece = np.exp(-(C/w)-(betaDeltaOverNu/(2*w**2))-np.sqrt(B)*C*np.arctan(np.sqrt(B)*w))
            powerlaw_piece = w**(-B*betaDeltaOverNu)*(1+B*w**2)**(B*betaDeltaOverNu/2)
            return etaScale*exponential_piece*powerlaw_piece
        else:
            print("functional type not recognized, using 'wellbehaved' form")
            return etaScale*np.exp(-B*C*w-betaDeltaOverNu/w)*w**(-B*betaDeltaOverNu+C)

