from rfim2d import scaling

r_list = [1.,2.,3.]

types = ['power law', 'truncated', 'well-behaved', 'pitchfork']


def test_Sigma_functional_form():
    for t in types:
        text = scaling.Sigma_functional_form(func_type=t)
        print(text)
    return


def test_eta_functional_form():
    for t in types:
        text = scaling.eta_functional_form(func_type=t)
        print(text)
    return

def test_Sigma_func():
    for t in types:
        if t == 'power law':
            param_dict = dict([('rScale',1.), ('rc',0.), ('sScale',1.), ('sigma', 1.)])
            Sigma = scaling.Sigma_func(r_list, param_dict, func_type=t)
        else:
            param_dict = dict([('rScale',1.), ('rc',0.), ('sScale',1.), ('df', 2.), ('B',1.), ('C',1.)])
            Sigma = scaling.Sigma_func(r_list, param_dict, func_type=t)
        print(Sigma)
    return
