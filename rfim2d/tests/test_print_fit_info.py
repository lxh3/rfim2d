import rfim2d
from rfim2d import print_fit_info

def test_print_A_fit_info():
    num_curves = 1
    params = [1.0, 1.0, 1.0]
    print_fit_info.print_A_fit_info(num_curves, params)
    
def test_print_dMdh_fit_info():
    num_curves = 1
    params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    print_fit_info.print_dMdh_fit_info(num_curves, params)
    
def test_print_Sigma_fit_info():
    params = [1.0, 1.0, 1.0, 1.0]
    print_fit_info.print_Sigma_fit_info(params)

def test_print_eta_fit_info():
    params = [1.0, 1.0, 1.0, 1.0, 1.0]
    print_fit_info.print_eta_fit_info(params)

def test_print_joint_constants():
    params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    print_fit_info.print_joint_fit_info(params, True)
