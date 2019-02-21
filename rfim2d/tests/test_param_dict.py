from rfim2d import param_dict

key_dict = {
    'A': ['Sigma', 'a', 'b'],
    'dMdh': ['hMax', 'eta', 'a', 'b', 'c'],
    'joint': ['rScale', 'rc', 'sScale', 'etaScale', 'df',
              'lambdaH', 'B', 'C', 'F'],
    'Sigma': ['rScale', 'rc', 'sScale', 'df', 'B', 'C'],
    'eta': ['rScale', 'rc', 'etaScale', 'lambdaH', 'B', 'F']
}


powerlaw_key_dict = {
    'joint': ['rScale', 'rc', 'sScale', 'etaScale', 'sigma', 'betaDelta'],
    'Sigma': ['rScale', 'rc', 'sScale', 'sigma'],
    'eta': ['rScale', 'rc', 'etaScale', 'betaDelta']
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
    assert param_dict.get_keys('A', func_type='power law') == -1
    keys3 = param_dict.get_keys('Sigma')
    keys4 = param_dict.get_keys('Sigma', func_type='power law')
    print(str(keys1)+str(keys3)+str(keys4))


def test_separate_params():
    keys = param_dict.get_keys('joint')
    values = [1. for i in range(len(keys))]
    params = param_dict.join_dict(keys,values)
    pS, pe = param_dict.separate_params(params)
    return pS, pe

def test_generate_and_split_dict():
    params = [1.0, 1.0]
    keys = ['A', 'B', 'C']
    fixed_dict = dict([('C', 0.)])
    new_dict = param_dict.generate_dict_with_fixed_params(params, keys, fixed_dict)
    vals = param_dict.split_dict_with_fixed_params(new_dict, fixed_dict) 
