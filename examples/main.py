import os
import numpy as np

import rfim2d
from rfim2d import save_and_load, fitting, plotting, scaling, errors

import matplotlib.style as style
style.use('seaborn-paper')

#### CREATE SAVES FOLDER ####

directory = 'saves/'
if not os.path.exists(directory):
    os.makedirs(directory)

#### AREA WEIGHTED SIZE DISTRIBUTION ####

# load data
r,s,A,sA = save_and_load.load_svA()
try:
    colors = save_and_load.load_func('colors_r.pkl.gz')
    print('Loading saved color choices')
except FileNotFoundError:
    num_colors = len(r)
    colors = plotting.generate_colors(num_colors)

# plot s vs sA
data = [r, s, sA]
labels = [r'$s$',r'$sA$']
logscale=[True,True]
Range = [[2e-1,1e5],[1e-3,1e0]]
figure_name = 'saves/A.png'
plotting.scatter(data, labels, logscale=logscale, Range=Range, figure_name=figure_name, colors=colors)

# fit scaling function
args = [s, sA]
params, err = fitting.fit_As_Scaling(args, filename='saves/As_fit_params')

# plot fits
constant_given_r = params['Sigma']
constant = [params['a'], params['b']]

data = [r, s, sA]
function = scaling.As_Scaling
labels = [r'$s$',r'$sA$']
figure_name = 'saves/A_fits.png'
plotting.scatter_vs_function(data, function, labels, constant, constant_given_r = constant_given_r, logscale=logscale, Range=Range, figure_name=figure_name, colors=colors)

# plot scaling collapse
function = scaling.As_Collapse
labels = [r'$s/\Sigma$',r'$sA$']
Range = [[1e-4,10],[1e-3,1e0]]
figure_name = 'saves/A_collapse.png'
plotting.collapse(data, function, labels, constant, constant_given_r, logscale=logscale, Range=Range, figure_name=figure_name, colors=colors)

#### DERIVATIVE OF THE MAGNETIZATION WITH RESPECT TO THE FIELD ####

# load data
r,h,dMdh = save_and_load.load_hvdMdh()

# plot h vs dMdh
data = [r,h,dMdh]
labels = [r'$h$', r'$dM/dh$']
logscale = [False,True]
Range = [[-3,4], [1e-2,30]]
figure_name = 'saves/dMdh.png'
plotting.scatter(data, labels, logscale=logscale, Range=Range, figure_name=figure_name, colors=colors)

# fit scaling function
args = [h,dMdh]
params, err = fitting.fit_dMdh_Scaling(args, filename='saves/dMdh_fit_params')

# plot fits
constant_given_r = np.asarray([params['hMax'], params['eta']]).T
constant = [params['a'], params['b'], params['c'], params['d']]

function = scaling.dMdh_Scaling
labels = [r'$h$', r'$dM/dH$']
Range = [[-1,3],[1e-2,30]]
figure_name = 'saves/dMdh_fits.png'
plotting.scatter_vs_function(data, function, labels, constant, constant_given_r = constant_given_r, logscale=logscale, Range=Range, figure_name=figure_name, colors=colors)

# plot scaling collapse
function = scaling.dMdh_Collapse
labels = [r'$(h-h_{max})/\eta$', r'$\eta dM/dH$']
logscale=[False,True]
Range = [[-5,5],[1e-2,2]]
figure_name = 'saves/dMdh_collapse.png'
plotting.collapse(data, function, labels, constant, constant_given_r, logscale=logscale, Range=Range, figure_name=figure_name, colors=colors)

#### NONLINEAR SCALING VARIABLES: SIGMA(R) AND ETA(R) ####

# fit Sigma and eta jointly
r,Sigma = fitting.get_Sigma()
r,eta = fitting.get_eta()
args = [r, Sigma, r, eta]
params, err = fitting.joint_fit(args)
params_Sigma, params_eta, prior = fitting.divvy_params(params, True, 'wellbehaved')

# plot fits
labels = [r'$r$',r'$\Sigma(r)$']
logscale = [False,True]
figure_name = 'saves/Sigma_fit.png'
loc = 'upper right'
plotting.compare([r,Sigma], scaling.Sigma_func, labels, params_Sigma, logscale=logscale, loc=loc, figure_name=figure_name, types='wellbehaved')
labels = [r'$r$',r'$\eta(r)$']
logscale = [False,True]
figure_name = 'saves/eta_fit.png'
loc = 'lower right'
plotting.compare([r,eta], scaling.eta_func, labels, params_eta, logscale=logscale, loc=loc, figure_name=figure_name, types='wellbehaved')

#### ERROR BARS ####

# calculate and save the error bars
figure_names = ['saves/Sigma_params_std.png','saves/eta_params_std.png']
data = errors.fit_and_plot_errors(figure_names=figure_names)
save_and_load.save_func('saves/Sigma_and_eta_params_std.pkl.gz',data)

#### COMPARISON WITH OTHER FORMS ####

# fit Sigma and eta with each form considered: powerlaw, simple, wellbehaved, and pitchfork
types = ['powerlaw', 'simple', 'wellbehaved', 'pitchfork']
try:
    colors = save_and_load.load_func('colors_types.pkl.gz')
    print('Loading saved color choices')
except FileNotFoundError:
    num_colors = len(types)
    colors = plotting.generate_colors(num_colors)
params_Sigma = []
params_eta = []
for t in types:
    params, err = fitting.joint_fit(args, func_type=t)
    pS, pe, prior = fitting.divvy_params(params, True, func_type=t)
    params_Sigma.append(pS)
    params_eta.append(pe)

# plot and save a figure comparing the various forms
data = [r,Sigma]
labels = [r'$r$',r'$\Sigma(r)$']
logscale = [False, True]
figure_name = 'saves/Sigma_comparison.png'
loc = 'upper right'
plotting.compare(data, scaling.Sigma_func, labels, params_Sigma, logscale=logscale, loc=loc, figure_name=figure_name, colors=colors)
data = [r,eta]
labels = [r'$r$',r'$\eta(r)$']
logscale = [False, True]
figure_name = 'saves/eta_comparison.png'
loc = 'lower right'
plotting.compare(data, scaling.eta_func, labels, params_eta, logscale=logscale, loc=loc, figure_name=figure_name, colors=colors)

