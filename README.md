Package for performs scaling collapses of 2D NE-RFIM simulation data

Author: Lorien X. Hayden
Date: 11-25-2018

Install using 'pip install rfim2d' 
(not hosted on pypi yet)

rfim2d/
	errors.py - calculates error bars
	fitting.py - performs fits
	param_dict.py - manipulates parameter dictionaries
	plotting.py - plots results
	print_fit_info.py - prints scaling function form and best fit parameter values
	residuals.py - defines residual functions
	save_and_load.py - functions to save and load data
	scaling.py - scaling forms

rfim2d/data/
	svA.pkl.gz - [r, s, A, As] = [disorder, size, area weighted avalanche size distribution A(s), size x A(s)]
	hvdMdh.pkl.gz - [r, h, dMdh] = [disorder, field, change in magnetization with field dMdh(h)]

rfim2d/tests/
	contains tests for each of the script files in rfim2d/ - not complete

examples/
	main.py - use examples, produces figures from text - not complete
