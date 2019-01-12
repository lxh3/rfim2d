from rfim2d import save_and_load, fitting


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
    params, err = fitting.Sigma_fit(args,
                                    func_type='simple',
                                    show_params=False)
    params, err = fitting.Sigma_fit(args,
                                    func_type='powerlaw',
                                    show_params=False)
    params, err = fitting.Sigma_fit(args,
                                    func_type='pitchfork',
                                    show_params=False)
    return


def test_Sigma_fit_variable_sigmaNu():
    r, params = test_fit_As()
    Sigma = params['Sigma']
    args = [r, Sigma]
    params, err = fitting.Sigma_fit(args,
                                    sigmaNu_fixed=False,
                                    show_params=False)
    params, err = fitting.Sigma_fit(args,
                                    sigmaNu_fixed=False,
                                    func_type='simple',
                                    show_params=False)
    params, err = fitting.Sigma_fit(args,
                                    sigmaNu_fixed=False,
                                    func_type='powerlaw',
                                    show_params=False)
    params, err = fitting.Sigma_fit(args,
                                    sigmaNu_fixed=False,
                                    func_type='pitchfork',
                                    show_params=False)
    return


def test_eta_fit():
    r, params = test_fit_dMdh()
    eta = params['eta']
    args = [r, eta]
    params, err = fitting.eta_fit(args, show_params=False)
    params, err = fitting.eta_fit(args,
                                  func_type='simple',
                                  show_params=False)
    params, err = fitting.eta_fit(args,
                                  func_type='powerlaw',
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
                                    func_type='simple',
                                    show_params=False)
    params, err = fitting.joint_fit(args,
                                    func_type='powerlaw',
                                    show_params=False)
    params, err = fitting.joint_fit(args,
                                    func_type='pitchfork',
                                    show_params=False)
    return


def test_joint_fits_variable_sigmaNu():
    r, params = test_fit_As()
    Sigma = params['Sigma']
    r, params = test_fit_dMdh()
    eta = params['eta']
    args = [r, Sigma, r, eta]
    params, err = fitting.joint_fit(args, sigmaNu_fixed=False,
                                    show_params=False)
    params, err = fitting.joint_fit(args, sigmaNu_fixed=False,
                                    func_type='simple',
                                    show_params=False)
    params, err = fitting.joint_fit(args, sigmaNu_fixed=False,
                                    func_type='powerlaw',
                                    show_params=False)
    params, err = fitting.joint_fit(args, sigmaNu_fixed=False,
                                    func_type='pitchfork',
                                    show_params=False)
    return


def test_fit_all():
    pA, pM, pS, pe = fitting.perform_all_fits(show_params=False)
