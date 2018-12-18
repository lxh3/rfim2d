import rfim2d 
from rfim2d import scaling

types = ['powerlaw', 'simple', 'wellbehaved', 'pitchfork']

def test_Sigma_functional_form():
    for t in types:
        text = scaling.Sigma_functional_form(func_type=t)
    return

def test_eta_functional_form():
    for t in types:
        text = scaling.eta_functional_form(func_type=t)
    return
