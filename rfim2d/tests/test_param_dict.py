from rfim2d import param_dict

key_dict = {
    'A': ['Sigma', 'a', 'b'],
    'dMdh': ['hMax', 'eta', 'a', 'b', 'c'],
    'joint': ['rScale', 'sScale', 'etaScale', 'sigmaNu',
              'betaDeltaOverNu', 'B', 'C', 'F'],
    'Sigma': ['rScale', 'sScale', 'sigmaNu', 'B', 'F'],
    'eta': ['rScale', 'etaScale', 'betaDeltaOverNu', 'B', 'C']
}


powerlaw_key_dict = {
    'joint': ['rScale', 'sScale', 'etaScale', 'sigma', 'betaDelta', 'rc'],
    'Sigma': ['rScale', 'sScale', 'sigma', 'rc'],
    'eta': ['rScale', 'etaScale', 'betaDelta', 'rc']
}


def test_split_dict():
    adict = {'one': 1, 'two': 2}
    keys, values = param_dict.split_dict(adict)
    assert keys == ['one', 'two']
    assert values == [1, 2]
    assert param_dict.split_dict('test') == -1


def test_joint_dict():
    keys = ['one', 'two']
    values = [1, 2]
    values_bad = [1, 2, 3]
    assert isinstance(param_dict.join_dict(keys, values), dict)
    assert param_dict.join_dict(keys, values_bad) == -1


def test_get_keys():
    keys1 = param_dict.get_keys('A')
    assert param_dict.get_keys('A', func_type='powerlaw') == -1
    keys3 = param_dict.get_keys('Sigma')
    keys4 = param_dict.get_keys('Sigma', func_type='powerlaw')
    print(str(keys1)+str(keys3)+str(keys4))


def test_ensure_dict():
    params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    sigmaNu_fixed = True
    params = param_dict.ensure_dict(params, key_dict, sigmaNu_fixed)
    assert isinstance(params, dict)
    sigmaNu_fixed = False
    params = param_dict.ensure_dict(params, key_dict, sigmaNu_fixed)
    assert isinstance(params, dict)
    return params


def test_separate_params():
    params = test_ensure_dict()
    pS, pe = param_dict.separate_params(params, key_dict)
    return pS, pe


def test_format_for_residual():
    pS, pe = test_separate_params()
    pS2, pe2, prior = param_dict.format_for_residual(pS, pe, 'wellbehaved')
    pS3, pe3, prior = param_dict.format_for_residual(pS, pe, 'powerlaw')


def test_divvy_params():
    params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    pS, pe, prior = param_dict.divvy_params(params,
                                            sigmaNu_fixed=True,
                                            func_type='simple')
    params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    pS, pe, prior = param_dict.divvy_params(params,
                                            sigmaNu_fixed=True,
                                            func_type='powerlaw')
