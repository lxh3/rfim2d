from rfim2d import save_and_load, fitting

A_params = {'a': 1.0, 'b': 1.0}
dMdh_params = {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0}
Sigma_powerlaw_params = {'rScale': 1.0, 'sScale': 1.0,
                         'sigma': 1.0, 'rc': 1.0}
Sigma_params = {'rScale': 1.0, 'sScale': 1.0, 'df': 1.0,
                'B': 1.0, 'C': 1.0}
eta_powerlaw_params = {'rScale': 1.0, 'etaScale': 1.0,
                       'betaDelta': 1.0, 'rc': 1.0}
eta_params = {'rScale': 1.0, 'etaScale': 1.0,
              'lambdaH': 1.0, 'B': 1.0, 'F': 1.0}
joint_powerlaw_params = {'rScale': 1.0, 'sScale': 1.0, 'etaScale': 1.0,
                         'sigma': 1.0, 'betaDelta': 1.0, 'rc': 1.0}
joint_params = {'rScale': 1.0, 'sScale': 1.0, 'etaScale': 1.0,
                'df': 1.0, 'lambdaH': 1.0, 'B': 1.0,
                'C': 1.0, 'F': 1.0}

def test_fit_As():
    r, s, A, As = save_and_load.load_svA()
    args = [s, As]
    params, err = fitting.fit_As_Scaling(args, show_params=False)
    return r, params


def test_fit_dMdh():
    r, h, dMdh = save_and_load.load_hvdMdh()
    args = [h, dMdh]
    params, err = fitting.fit_dMdh_Scaling(args, show_params=False)
    return r, params


def test_Sigma_fit():
    r, params = test_fit_As()
    Sigma = params['Sigma']
    args = [r, Sigma]
    params, err = fitting.Sigma_fit(args, show_params=False)
    params, err = fitting.Sigma_fit(args, func_type='truncated', 
                                    show_params=False)
    params, err = fitting.Sigma_fit(args, func_type='power law',
                                    show_params=False)
    params, err = fitting.Sigma_fit(args, func_type='pitchfork',
                                    show_params=False)
    return


def test_eta_fit():
    r, params = test_fit_dMdh()
    eta = params['eta']
    args = [r, eta]
    params, err = fitting.eta_fit(args, show_params=False)
    params, err = fitting.eta_fit(args,
                                  func_type='truncated',
                                  show_params=False)
    params, err = fitting.eta_fit(args,
                                  func_type='power law',
                                  show_params=False)
    params, err = fitting.eta_fit(args,
                                  func_type='pitchfork',
                                  show_params=False)
    return


def test_joint_fits():
    r, params = test_fit_As()
    Sigma = params['Sigma']
    r, params = test_fit_dMdh()
    eta = params['eta']
    args = [r, Sigma, r, eta]
    params, err = fitting.joint_fit(args, show_params=False)
    params, err = fitting.joint_fit(args,
                                    func_type='truncated',
                                    show_params=False)
    params, err = fitting.joint_fit(args,
                                    func_type='power law',
                                    show_params=False)
    params, err = fitting.joint_fit(args,
                                    func_type='pitchfork',
                                    show_params=False)
    return


def test_fit_all():
    pA, pM, pS, pe = fitting.perform_all_fits(show_params=False)
