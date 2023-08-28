from src.parser import *

# Data folders
output_folder = 'processed'
data_folder = 'data'

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'SWaT': [(0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'WADI': [(0.99, 1), (0.999, 1)],
	}
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
		'SMD': 0.0001, 
		'SWaT': 0.008, 
		'SMAP': 0.001, 
		'MSL': 0.002, 
		'WADI': 0.0001, 
		'UCR': 0.006, 
	}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
		'SMD': (98, 2000),
		'SWaT': (95, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'WADI': (99, 1200),
		'UCR': (98, 2),
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9