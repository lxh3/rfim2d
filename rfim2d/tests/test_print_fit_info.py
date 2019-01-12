from rfim2d import print_fit_info


show = False


A_params = {'a': 1.0, 'b': 1.0}
dMdh_params = {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0}
Sigma_powerlaw_params = {'rScale': 1.0, 'sScale': 1.0,
                         'sigma': 1.0, 'rc': 1.0}
Sigma_params = {'rScale': 1.0, 'sScale': 1.0, 'sigmaNu': 1.0,
                'B': 1.0, 'F': 1.0}
eta_powerlaw_params = {'rScale': 1.0, 'etaScale': 1.0,
                       'betaDelta': 1.0, 'rc': 1.0}
eta_params = {'rScale': 1.0, 'etaScale': 1.0,
              'betaDeltaOverNu': 1.0, 'B': 1.0, 'C': 1.0}
joint_powerlaw_params = {'rScale': 1.0, 'sScale': 1.0, 'etaScale': 1.0,
                         'sigma': 1.0, 'betaDelta': 1.0, 'rc': 1.0}
joint_params = {'rScale': 1.0, 'sScale': 1.0, 'etaScale': 1.0,
                'sigmaNu': 1.0, 'betaDeltaOverNu': 1.0, 'B': 1.0,
                'C': 1.0, 'F': 1.0}


def test_print_all():
    print_fit_info.print_fit_info(A_params, 'A', show=False)
    print_fit_info.print_fit_info(dMdh_params, 'dMdh', show=False)
    print_fit_info.print_fit_info(Sigma_powerlaw_params, 'Sigma',
                                  func_type='powerlaw', show=False)
    print_fit_info.print_fit_info(Sigma_params, 'Sigma', show=False)
    print_fit_info.print_fit_info(Sigma_params, 'Sigma',
                                  func_type='pitchfork', show=False)
    print_fit_info.print_fit_info(eta_powerlaw_params, 'eta',
                                  func_type='powerlaw', show=False)
    print_fit_info.print_fit_info(eta_params, 'eta', show=False)
    print_fit_info.print_fit_info(eta_params, 'eta',
                                  func_type='pitchfork', show=False)
    print_fit_info.print_fit_info(joint_powerlaw_params, 'joint',
                                  func_type='powerlaw', show=False)
    print_fit_info.print_fit_info(joint_params, 'joint', show=False)
    return
