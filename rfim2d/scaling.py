import numpy as np

from scipy.special import gamma, gammaincc

def As_Scaling(s, constant_with_r, constant):
    """
    Assumed form of the area weighted size distribution (A) times the avalanche size (s) 
    as a function of the avalanche size, Sigma(r), and unknown constants a and b
    """
    Sigma = constant_with_r
    a, b = constant

    sscaled = s/Sigma #scaling variable s/Sigma(r)
    As = sscaled**a*np.exp(-sscaled**b) 
    AsNorm = gamma(a/b)*gammaincc(a/b,Sigma**(-2*b))/b #Normalization
    return As/AsNorm

def As_Collapse(s, As, constant_with_r, constant):
    """
    Function to obtain the values to plot in order to collapse A
    """
    xscaled = s/constant_with_r
    yscaled = As
    yscaled_from_function = As_Scaling(s, constant_with_r, constant)
    return xscaled, yscaled, yscaled_from_function

def Sigma_func(r_list, constant, simple=False, scaled=False):
    """
    Nonlinear scaling variable function determined from 
    the flow equations of the disorder, w, and avalanche size, s
    Input:
        r_list - list of r values
        constant - constants on which Sigma(r) is dependent
        simple - flag for whether to use the Sigma derived from dw/dl = w^2+B w^3
        powerlaw - flag for whether to use the 
        scaled - flag for whether to use the Sigma derived from the simplest dw/dl
                 scaled for comparison with complicated form at small r
    Output:
        Sigma(r) values for the given inputs
    """
    rScale, sScale, sigmaNu, B, F = constant
    r = np.array(r_list)
    w = r/rScale
    if simple:
        Sigma = abs(sScale)*np.exp(1/(w*sigmaNu))*(B+(1/w))**(F-(B/sigmaNu))
        if scaled:
            Sigma = np.exp(B/sigmaNu)*Sigma
        return Sigma 
    else:
        return abs(sScale)*np.exp(B*F*w + 1/(w*sigmaNu))*w**((B/sigmaNu)-F)

def dMdh_Scaling(h, constant_with_r, constant):
    """
    Assumed form of the derivative of the magnetization (M) with respect to the field (h)
    as a function of the field and unknown constants a, b, c and power
    """
    hmax, eta = constant_with_r
    a, b, c, d = constant

    hscaled = (h-hmax)/eta #scaling variable (h-hmax(r))/eta(r)
    dMdh = np.exp(-hscaled**2/(a+b*hscaled+c*hscaled**2)**(d/2.))
    dh = 0.01
    dMdhNorm = dh*sum(dMdh)/2.#Normalization
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


def eta_func(r_list, constant, simple=False, **kwargs):
    """
    Nonlinear scaling variable function determined from
    the flow equations for the disorder, w, and the field, h
    """
    rScale,etaScale,betaDeltaOverNu,B,C = constant
    r = np.array(r_list)
    w = np.array(r)/rScale
    if simple:
        return etaScale*np.exp(-betaDeltaOverNu/w)*(B+(1/w))**(-C+B*betaDeltaOverNu)
    else:
        return etaScale*np.exp(-B*C*w-betaDeltaOverNu/w)*w**(-B*betaDeltaOverNu+C)
