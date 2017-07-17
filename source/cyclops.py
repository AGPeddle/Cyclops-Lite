#!/usr/bin/env/ python3
"""
This code implements the Asymptotic Parallel in Time (APinT) algorithm
for the 1-D rotating shallow water equations, following Haut and Wingate (2014).

Control and parameterisation of the code is through the cyclops_control library.

The structure of the program is based on the Fourier Spectral code Morgawr, by AGP. A significant
amount of the numerical code is courtesy of Dr. Terry Haut of LANL.

Algorithm
---------
1) Set up relevant (Python) objects for computation to progress (spectral_toolbox, control, exponential_integrator)

2) Call solver method for timestepping

3) Output writing

| Author: Adam G. Peddle, Terry Haut
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import time
import sys
import os
import pickle

from spectral_toolbox_1D import SpectralToolbox
from RSWE_exponential_integrator import ExponentialIntegrator
import RSWE_direct
import cyclops_base
import numpy as np
import cyclops_control

def APinT_solver(control, st, expInt, u_init):
    """
    Implements the Asymptotic Parallel in Time algorithm.

    This methods proceeds by performing a coarse approximation using the coarse propagator
    which is then refined (in theory, in parallel) by the fine propagator. It differs from
    standard parareal methods by the averaging over the fast waves which is performed in
    the coarse timestep. The fine timestep solves the full equations to the desired level
    of accuracy.

    Implementation follows:

    .. math:: U_{n+1}^{k+1} = G(U_{n}^{k+1}) + F(U_{n}^{k}) - G(U_{n}^{k})

    where G refers to the coarse propagator and F to the fine propagator. n is the timestep
    k is the current iteration number. Converges to the accuracy of the fine propgator.

    **Parameters**

    - `control` : a control object
    - `st` : the spectral toolbox object
    - `expInt` : an exponential integrator object
    - `u_init` : the initial conditions in Fourier space, np.array(3,Nx), (u,v,h)

    **Returns**

    - `errs` : A tuple containing lists of the L_infty and L_2 errors vs iteration, respectively
    - `U_hat_new` : The solution after convergence (np.array(Nt, 3, Nx))

    **See Also**

    coarse_propagator, fine_propagator

    """
    # U_hat_mac contains the solution at the completion of the
    # coarse parareal timesteps.
    # U_hat_mic contains the solution at the completion of the fine
    # parareal timesteps, but discards the information in between
    # (i.e. matches the timesteps of the coarse solution)
    U_hat_mac = np.zeros((control['Nt'], 3, control['Nx']), dtype = 'complex')
    U_hat_mic = np.zeros((control['Nt'], 3, control['Nx']), dtype = 'complex')
    U_hat_new = np.zeros((control['Nt'], 3, control['Nx']), dtype = 'complex')
    U_hat_old = np.zeros((control['Nt'], 3, control['Nx']), dtype = 'complex')

    # U_hat_old contains the solution at the previous iteration for
    # use in convergence testing
    errors = [[] for _ in range(control['Nt'])]

    # Create initial condition
    U_hat_new[0,:,:] = u_init

    for k in range(3):
        U_hat_new[0,k,:] = st.forward_fft(U_hat_new[0,k,:])

    U_hat_old[0,:,:] = U_hat_new[0,:,:]

    # Compute first parareal level here
    start = time.time()
    for j in range(control['Nt']-1):
        # First parareal level by coarse timestep in serial only
        U_hat_new[j+1, :, :] = RSWE_direct.solve(control, expInt, st,
                                                 U_hat_new[j, :, :],
                                                 solver = 'coarse_propagator',
                                                 realspace = False)

        U_hat_old[j+1,:,:] = U_hat_new[j+1, :, :]

    end = time.time()
    print("First APinT level completed in {:.8f} seconds".format(end-start))

    # Further parareal levels computed here
    iterative_error = 100000000000000.

    L_inf_buffer = []
    L_2_buffer = []

    while iterative_error > control['conv_tol']:
        start = time.time()
        L_infty_err = 0.
        L_2_err = 0.

        U_hat_new = np.zeros((control['Nt'], 3, control['Nx']), dtype = 'complex')
        U_hat_new[0,:,:] = U_hat_old[0,:,:]

        for j in range(control['Nt']-1):  # Loop over timesteps
            # Compute coarse and fine timesteps (parallel-isable)
            U_hat_mac[j+1, :, :] = RSWE_direct.solve(control, expInt, st,
                                                     U_hat_old[j, :, :],
                                                     solver = 'coarse_propagator',
                                                     realspace = False)

            U_hat_mic[j+1, :, :] = RSWE_direct.solve(control, expInt, st,
                                                     U_hat_old[j, :, :],
                                                     solver = 'fine_propagator',
                                                     realspace = False)

            # Compute and apply correction (serial)
            U_hat_new[j+1, :, :] = RSWE_direct.solve(control, expInt, st,
                                                     U_hat_new[j, :, :],
                                                     solver = 'coarse_propagator',
                                                     realspace = False)

            U_hat_new[j+1, :, :] = U_hat_new[j+1, :,:] + (U_hat_mic[j+1, :,:] - U_hat_mac[j+1, :,:])

            # L_inf, L_2
            error_iteration = cyclops_base.compute_errors(U_hat_old[j+1,:,:], U_hat_new[j+1,:,:])
            L_infty_err = max(L_infty_err, error_iteration[0])
            L_2_err = max(L_2_err, error_iteration[1])

        L_inf_buffer.append(L_infty_err)
        L_2_buffer.append(L_2_err)

        # Perform convergence checks (iterative error)
        U_hat_old[:, :, :] = U_hat_new[:, :, :].copy()  #Overwrite previous solution for convergence tests
        iterative_error_old = iterative_error
        iterative_error = L_infty_err

        end = time.time()
        print("APinT level {:>2} completed in {:.8f} seconds".format(k,end-start))
        print("L_infty norm = {:.6e}".format(iterative_error))

        if iterative_error > iterative_error_old:
            print('Possible Numerical Instability Detected.')

    print("APinT Computation Complete in {:>2} iterations.".format(k))

    return (L_inf_buffer, L_2_buffer), U_hat_new


if __name__ == "__main__":
    control = cyclops_control.setup_control(sys.argv[1:])
    if 'working_dir' in control: os.chdir(control['working_dir'])

    # Local parameterisations:
    if control['outFileStem'] is None: control['outFileStem'] = ''
    conv_tol = control['conv_tol']
    control['final_time'] = control['coarse_timestep']  # For generalisabilty of rswe_direct
    control['solver'] = None  # Idiot-proofing

    # Initialise spectral toolbox object
    st = SpectralToolbox(control['Nx'], control['Lx'])

    if control['HMM_T0'] is None:
        control['HMM_T0'] = control['coarse_timestep']/(control['epsilon']**0.2)
    control['HMM_M_bar'] = max(25, int(80*control['HMM_T0']))

    #Create exponential integrator object
    expInt = ExponentialIntegrator(control)

    # Set up initial (truth) field
    ICs = cyclops_base.h_init(control)

    # Set up spectral toolbox
    st = SpectralToolbox(control['Nx'], control['Lx'])

    # ICs should be in Fourier space for APinT algo as implemented
    for k in range(3):
        IC_hat = np.zeros((3, control['Nx']), dtype=complex)
        IC_hat[k,:] = st.forward_fft(ICs[k,:])

    # Kernel. Call to solver.
    errs, U_hat_new = APinT_solver(control, st, expInt, IC_hat)

    # Post-convergence, handle output
    for i in range(control['Nt']):
        for k in range(3):
            U_hat_new[i,k,:] = st.inverse_fft(U_hat_new[i,k,:])

    with open("{}_APinT.dat".format(control['outFileStem']), 'wb') as f:

        data = dict()
        data['u'] = np.real(U_hat_new[i, 0, :])
        data['v'] = np.real(U_hat_new[i, 1, :])
        data['h'] = np.real(U_hat_new[i, 2, :])

        data['L_infty_errs'] = errs[0]
        data['L_2_errs'] = errs[1]

        data['control'] = control

        pickle.dump(data, f)

