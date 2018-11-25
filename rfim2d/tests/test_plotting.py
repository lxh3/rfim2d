import os
import numpy as np

import rfim2d
from rfim2d import general_use, scaling, fitting, plotting


def test_plotting_svA():
    r,s,A,sA = general_use.load_svA()
    data = [r, s, sA]
    labels = [r'$s$',r'$sA$']
    logscale=[True,True]
    Range = [[2e-1,1e5],[1e-3,1e0]]
    plotting.plot_xy(data,labels,logscale=logscale, Range=Range)
    return

def test_plotting_hvdMdh():
    r,h,dMdh = general_use.load_hvdMdh()
    data = [r,h,dMdh]
    labels = [r'$h$', r'$dM/dh$']
    logscale = [False,True]
    Range = [[-3,4], [1e-2,30]]
    plotting.plot_xy(data,labels, logscale=logscale, Range=Range)
    return

def test_plotting_fits_svA():
    r,s,A,As = general_use.load_svA()
    data = [r, s, As]
    function = scaling.As_Scaling 
    labels = [r'$s$',r'$sA$']
    logscale=[True,True]
    args = [s, As]
    params, err = fitting.fit_As_Scaling(args)
    constant_given_r = params[:len(r)]
    constant = params[len(r):]
    Range = [[2e-1,1e5],[1e-3,1e0]]
    plotting.plot_xy_and_xfofx(data, function, labels, constant, constant_given_r = constant_given_r, logscale=logscale, Range=Range)
    return
   
def test_plotting_fits_hvdMdh():
    r, h, dMdh = general_use.load_hvdMdh()
    data = [r, h, dMdh]
    function = scaling.dMdh_Scaling
    labels = [r'$h$', r'$dM/dH$']
    logscale = [False,True]
    args = [h, dMdh]
    params, err = fitting.fit_dMdh_Scaling(args)
    constant_given_r = np.array(params[:2*len(r)]).reshape((2,len(r))).T
    constant = params[2*len(r):]
    Range = [[-1,3],[1e-2,30]]
    plotting.plot_xy_and_xfofx(data, function, labels, constant, constant_given_r = constant_given_r, logscale=logscale, Range=Range)
    return

def test_plotting_collapse_svA():
    r,s,A,As = general_use.load_svA()
    data = [r, s, As]
    function = scaling.As_Collapse
    labels = [r'$s/\Sigma$',r'$sA$']
    logscale=[True,True]
    args = [s, As]
    params, err = fitting.fit_As_Scaling(args)
    constant_given_r = params[:len(r)]
    constant = params[len(r):]
    Range = [[1e-4,10],[1e-3,1e0]]
    plotting.plot_collapse(data, function, labels, constant, constant_given_r, logscale=logscale, Range=Range)
    return

def test_plotting_collapse_hvdMdh():
    r,h,dMdh = general_use.load_hvdMdh()
    data = [r, h, dMdh]
    function = scaling.dMdh_Collapse
    labels = [r'$(h-h_{max})/\eta$', r'$\eta dM/dH$']
    logscale=[False,True]
    args = [h, dMdh]
    params, err = fitting.fit_dMdh_Scaling(args)
    constant_given_r = np.array(params[:2*len(r)]).reshape((2,len(r))).T
    constant = params[2*len(r):]
    Range = [[-5,5],[1e-2,2]]
    plotting.plot_collapse(data, function, labels, constant, constant_given_r, logscale=logscale, Range=Range)
    return

def test_plotting_joint_fits():
    # Get Sigma(r)
    r,s,A,As = general_use.load_svA()
    args = [s, As]
    params, err = fitting.fit_As_Scaling(args)
    Sigma = params[:len(r)]
    # Get eta(r)
    r,h,dMdh = general_use.load_hvdMdh()
    args = [h,dMdh]
    params, err = fitting.fit_dMdh_Scaling(args)
    eta = params[len(r):2*len(r)]
    # Perform joint fit
    priorWeight = 0.1
    args = [r, Sigma, eta, priorWeight]
    params, err = fitting.joint_fit(args)
    # Divvy up params between Sigma and eta
    sigmaNu = 0.5
    rScale, sScale, etaScale, betaDeltaOverNu, B, C, F = params
    params_Sigma = rScale, sScale, sigmaNu, B, F
    params_eta = rScale, etaScale, betaDeltaOverNu, B, C
    #Plot Sigma vs SigmaFit
    data = [None, r, Sigma]
    labels = [r'$r$',r'$\Sigma(r)$']
    logscale = [False,True]
    plotting.plot_xy_and_xfofx(data, scaling.Sigma_func, labels, params_Sigma, logscale=logscale)
    #Plot eta vs etaFit
    data = [None, r, eta]
    labels = [r'$r$',r'$\eta(r)$']
    logscale = [False,False]
    plotting.plot_xy_and_xfofx(data, scaling.eta_func, labels, params_eta, logscale=logscale)
    return

def test_plotting_joint_fits_variablesigmaNu():
    # Get Sigma(r)
    r,s,A,As = general_use.load_svA()
    args = [s, As]
    params, err = fitting.fit_As_Scaling(args)
    Sigma = params[:len(r)]
    # Get eta(r)
    r,h,dMdh = general_use.load_hvdMdh()
    args = [h,dMdh]
    params, err = fitting.fit_dMdh_Scaling(args)
    eta = params[len(r):2*len(r)]
    # Perform joint fit
    priorWeight = 0.1
    args = [r, Sigma, eta, priorWeight]
    params, err = fitting.joint_fit(args, sigmaNu_fixed=False)
    # Divvy up params between Sigma and eta
    rScale, sScale, etaScale, sigmaNu, betaDeltaOverNu, B, C, F = params
    params_Sigma = rScale, sScale, sigmaNu, B, F
    params_eta = rScale, etaScale, betaDeltaOverNu, B, C
    #Plot Sigma vs SigmaFit
    data = [None, r, Sigma]
    labels = [r'$r$',r'$\Sigma(r,sigmaNu)$']
    logscale = [False,True]
    plotting.plot_xy_and_xfofx(data, scaling.Sigma_func, labels, params_Sigma, logscale=logscale)
    #Plot eta vs etaFit
    data = [None, r, eta]
    labels = [r'$r$',r'$\eta(r,sigmaNu)$']
    logscale = [False,False]
    plotting.plot_xy_and_xfofx(data, scaling.eta_func, labels, params_eta, logscale=logscale)
    return

