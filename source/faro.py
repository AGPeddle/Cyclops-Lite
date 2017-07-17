#!/bin/python3

import sys
import os
import pickle
import cyclops_control
import numpy as np
import RSWE_direct
from EnKF import Ensemble_Kalman
import matplotlib.pyplot as plt
from RSWE_exponential_integrator import ExponentialIntegrator
from spectral_toolbox_1D import SpectralToolbox

control = cyclops_control.setup_control(sys.argv[1:])
if 'working_dir' in control: os.chdir(control['working_dir'])

with open('{}_assim_data.dat'.format(control['outFileStem']), 'rb') as f:
    assim_data = pickle.load(f)

    meas_locs = np.hstack((assim_data['x_meas'], control['Nx'] + assim_data['x_meas'], 2*control['Nx'] + assim_data['x_meas']))
    ts = assim_data['t']
    measurements = assim_data['u']
    ICs = assim_data['ICs']
    size = assim_data['size']

enKF = Ensemble_Kalman(meas_locs, 3*control['Nx'], size)

# Create exponential integrator:
expInt = ExponentialIntegrator(control)

# Initialise spectral toolbox object
st = SpectralToolbox(control['Nx'], control['Lx'])

x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])
U = np.zeros((size, 3, control['Nx']))
for k in range(size):
    U[k, 2, :] = ICs[k, 2, :]

alldata = np.zeros((len(ts)*(control['assim_cycle_length'] + 1), size, 3, control['Nx']))
vs = 0

for nt, T in enumerate(ts[1:]):
    print("Running timestep {} ending at {}".format(nt, T))
    control['final_time'] = (T - ts[nt])/control['assim_cycle_length']

    for ss in range(control['assim_cycle_length']):
        for n in range(size):
            if ss == 0: print("Running ensemble {}".format(n))
            U[n, :, :] = RSWE_direct.solve(control, expInt, st, U[n, :, :])
            alldata[vs, n, :, :] = U[n,:,:]
        vs += 1

    d = measurements[nt + 1, :, :].reshape(-1,1)
    A = U.reshape(size, -1)

    A = enKF.apply(np.transpose(A), d)
    A = np.transpose(A)
    U = A.reshape(size, 3, control['Nx'])
    alldata[vs,:,:,:] = U[:,:,:]
    vs += 1

with open('{}_preforecast.dat'.format(control['outFileStem']), 'wb') as f:
    pickle.dump(alldata, f)

