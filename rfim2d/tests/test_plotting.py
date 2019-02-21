import numpy as np

from rfim2d import save_and_load, scaling, fitting, plotting


r, s, A, As = save_and_load.load_svA()
r, h, dMdh = save_and_load.load_hvdMdh()


def test_plotting_svA():
    data = [r, s, As]
    labels = [r'$s$', r'$sA$']
    logscale = [True, True]
    Range = [[2e-1, 1e5], [1e-3, 1e0]]
    figure_name = None  # 'A.png'
    plotting.scatter(data, labels, logscale=logscale,
                     Range=Range, figure_name=figure_name,
                     show=False)
    return


def test_plotting_hvdMdh():
    data = [r, h, dMdh]
    labels = [r'$h$', r'$dM/dh$']
    logscale = [False, True]
    Range = [[-3, 4], [1e-2, 30]]
    figure_name = None  # 'dMdh.png'
    plotting.scatter(data, labels, logscale=logscale,
                     Range=Range, figure_name=figure_name,
                     show=False)
    return


def test_plotting_fits_svA():
    data = [r, s, As]
    function = scaling.As_Scaling
    labels = [r'$s$', r'$sA$']
    logscale = [True, True]
    args = [s, As]
    params, err = fitting.fit_As_Scaling(args, show_params=False)
    constant_given_r = params['Sigma']
    constant = [params['a'], params['b']]
    Range = [[2e-1, 1e5], [1e-3, 1e0]]
    figure_name = None  # 'A_fits.png'
    plotting.scatter_vs_function(data, function, labels, constant,
                                 constant_given_r=constant_given_r,
                                 logscale=logscale, Range=Range,
                                 figure_name=figure_name, show=False)
    return


def test_plotting_fits_hvdMdh():
    data = [r, h, dMdh]
    function = scaling.dMdh_Scaling
    labels = [r'$h$', r'$dM/dH$']
    logscale = [False, True]
    args = [h, dMdh]
    params, err = fitting.fit_dMdh_Scaling(args, show_params=False)
    constant_given_r = np.asarray([params['hMax'], params['eta']]).T
    constant = [params['a'], params['b'], params['c'], params['d']]
    Range = [[-1, 3], [1e-2, 30]]
    figure_name = None  # 'dMdh_fits.png'
    plotting.scatter_vs_function(data, function, labels, constant,
                                 constant_given_r=constant_given_r,
                                 logscale=logscale, Range=Range,
                                 figure_name=figure_name, show=False)
    return


def test_plotting_collapse_svA():
    data = [r, s, As]
    function = scaling.As_Collapse
    labels = [r'$s/\Sigma$', r'$sA$']
    logscale = [True, True]
    args = [s, As]
    params, err = fitting.fit_As_Scaling(args, show_params=False)
    constant_given_r = params['Sigma']
    constant = [params['a'], params['b']]
    Range = [[1e-4, 10], [1e-3, 1e0]]
    figure_name = None  # 'A_collapse.png'
    plotting.collapse(data, function, labels, constant,
                      constant_given_r, logscale=logscale,
                      Range=Range, figure_name=figure_name,
                      show=False)
    return


def test_plotting_collapse_hvdMdh():
    data = [r, h, dMdh]
    function = scaling.dMdh_Collapse
    labels = [r'$(h-h_{max})/\eta$', r'$\eta dM/dH$']
    logscale = [False, True]
    args = [h, dMdh]
    params, err = fitting.fit_dMdh_Scaling(args, show_params=False)
    constant_given_r = np.asarray([params['hMax'], params['eta']]).T
    constant = [params['a'], params['b'], params['c'], params['d']]
    Range = [[-5, 5], [1e-2, 2]]
    figure_name = None  # 'dMdh_collapse.png'
    plotting.collapse(data, function, labels, constant,
                      constant_given_r, logscale=logscale,
                      Range=Range, figure_name=figure_name,
                      show=False)
    return


def get_Sigma():
    args = [s, As]
    params, err = fitting.fit_As_Scaling(args, show_params=False)
    Sigma = params['Sigma']
    return Sigma


def get_eta():
    args = [h, dMdh]
    params, err = fitting.fit_dMdh_Scaling(args, show_params=False)
    eta = params['eta']
    return eta


def get_args():
    Sigma = get_Sigma()
    data_Sigma = [r, Sigma]
    eta = get_eta()
    data_eta = [r, eta]
    args = [r, Sigma, r, eta]
    return data_Sigma, data_eta, args


def perform_joint_fits(args, func_type='well-behaved'):
    params_A, params_dMdh, params_Sigma, params_eta = fitting.perform_all_fits(func_type=func_type, show_params=False)
    return params_Sigma, params_eta


def test_plotting_joint_fits():
    data_Sigma, data_eta, args = get_args()
    types = ['power law', 'truncated', 'well-behaved', 'pitchfork']
    for t in types:
        params_Sigma, params_eta = perform_joint_fits(args, func_type=t)
        labels = [r'$r$', r'$\Sigma(r)$']
        logscale = [False, True]
        figure_name = None  # 'Sigma_'+t+'.png'
        loc = 'upper right'
        plotting.compare(data_Sigma, scaling.Sigma_func, labels,
                         params_Sigma, logscale=logscale, loc=loc,
                         figure_name=figure_name, types=t, show=False)
        labels = [r'$r$', r'$\eta(r)$']
        logscale = [False, True]
        figure_name = None  # 'eta_'+t+'.png'
        loc = 'lower right'
        plotting.compare(data_eta, scaling.eta_func, labels,
                         params_eta, logscale=logscale, loc=loc,
                         figure_name=figure_name, types=t, show=False)
    return


def test_comparing_all_forms():
    data_Sigma, data_eta, args = get_args()
    types = ['power law', 'truncated', 'well-behaved', 'pitchfork']
    pS = []
    pe = []
    for t in types:
        params_Sigma, params_eta = perform_joint_fits(args, func_type=t)
        pS.append(params_Sigma)
        pe.append(params_eta)
    labels = [r'$r$', r'$\Sigma(r)$']
    logscale = [False, True]
    loc = 'upper right'
    plotting.compare(data_Sigma, scaling.Sigma_func, labels,
                     pS, logscale=logscale, loc=loc, show=False)
    labels = [r'$r$', r'$\eta(r)$']
    logscale = [False, True]
    loc = 'lower right'
    plotting.compare(data_eta, scaling.eta_func, labels,
                     pe, logscale=logscale, loc=loc, show=False)
    return
