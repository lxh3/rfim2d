from rfim2d import scaling


types = ['powerlaw', 'simple', 'wellbehaved', 'pitchfork']


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
