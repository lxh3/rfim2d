import rfim2d
from rfim2d import general_use, fitting

def test_fit_As():
    r,s,A,As = general_use.load_svA()
    args = [s, As]    
    params, err = fitting.fit_As_Scaling(args)
    return r, params

def test_fit_dMdh():
    r,h,dMdh = general_use.load_hvdMdh()
    args = [h,dMdh]
    params, err = fitting.fit_dMdh_Scaling(args)
    return r, params

def test_joint_fit():
    priorWeight = 0.1
    r, params = test_fit_As()
    Sigma = params[:len(r)]
    r, params = test_fit_dMdh()
    eta = params[len(r):2*len(r)]
    args = [r, Sigma, eta, priorWeight]
    params, err = fitting.joint_fit(args)
    return

def test_joint_fit_simple():
    priorWeight = 0.1
    r, params = test_fit_As()
    Sigma = params[:len(r)]
    r, params = test_fit_dMdh()
    eta = params[len(r):2*len(r)]
    args = [r, Sigma, eta, priorWeight]
    params, err = fitting.joint_fit(args, simple=True)
    return
