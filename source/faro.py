#!/usr/bin/env/ python3
"""
This script provides a skeleton of an implementation for assimilating data
into the 1-D rotating shallow water equations via the Ensemble Kalman filter
due to Evensen (2003). Simulated measured data based on a fine numerical solve
is obtained in the faro_datamaker script.

Both the fine (full) and coarse (averaged) timestepping methods are implemented
and available to the user. Output is in a pickled dict.

| Author: Adam G. Peddle
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import sys
import os
import pickle
import cyclops_control
import numpy as np
import RSWE_direct
from EnKF import Ensemble_Kalman
from RSWE_exponential_integrator import ExponentialIntegrator
from spectral_toolbox_1D import SpectralToolbox

if __name__ == "__main__":
    control = cyclops_control.setup_control(sys.argv[1:])
    if 'working_dir' in control: os.chdir(control['working_dir'])

    # Set up data from datamaker
    with open('{}_assim_data.dat'.format(control['outFileStem']), 'rb') as f:
        assim_data = pickle.load(f)

        meas_locs = np.hstack((assim_data['x_meas'], control['Nx'] + assim_data['x_meas'], 2*control['Nx'] + assim_data['x_meas']))
        ts = assim_data['t']
        measurements = assim_data['u']
        ICs = assim_data['ICs']
        size = assim_data['size']

    # Initialise the Ensemble Kalman filter, precomputing at least some of it
    enKF = Ensemble_Kalman(meas_locs, 3*control['Nx'], size)

    # Create exponential integrator:
    expInt = ExponentialIntegrator(control)

    # Initialise spectral toolbox object
    st = SpectralToolbox(control['Nx'], control['Lx'])

    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])
    U = np.zeros((size, 3, control['Nx']))
    for k in range(size):
        U[k, 2, :] = ICs[k, 2, :]

    # Big dirty container for all timesteps and ensemble members
    alldata = np.zeros((len(ts)*(control['assim_cycle_length'] + 1), size, 3, control['Nx']))
    vs = 0

    # Loop over all assimilation cycles
    for nt, T in enumerate(ts[1:]):
        print("Running timestep {} ending at {}".format(nt, T))
        control['final_time'] = (T - ts[nt])/control['assim_cycle_length']

        # Loop over coarse timesteps
        # These loops are the 'prediction' step
        for ss in range(control['assim_cycle_length']):
            # Loop over ensemble members
            for n in range(size):
                if ss == 0: print("Running ensemble {}".format(n))
                U[n, :, :] = RSWE_direct.solve(control, expInt, st, U[n, :, :])
                alldata[vs, n, :, :] = U[n,:,:]
            vs += 1

        # Assimilation of measured data via EnKF
        # This bit is the 'update' step
        d = measurements[nt + 1, :, :].reshape(-1,1)
        A = U.reshape(size, -1)

        A = enKF.apply(np.transpose(A), d)
        A = np.transpose(A)
        U = A.reshape(size, 3, control['Nx'])
        alldata[vs,:,:,:] = U[:,:,:]
        vs += 1

    # Dump the data. The last timestep is a suitable initial condition
    # for a simulation.
    with open('{}_preforecast.dat'.format(control['outFileStem']), 'wb') as f:
        pickle.dump(alldata, f)

