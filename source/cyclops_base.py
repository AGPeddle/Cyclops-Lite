#!/usr/bin/env/ python3
"""
This code implements the solution to the Rotating Shallow Water Equations (RSWE). The
primary goal is to investigate Asymptotic Parallel-in-Time (APinT) methods.

The only input requirement for this code is the control dictionary, which is located at the top of the program.

The structure of the program is based on the Fourier Spectral code Morgawr, by AGP. A significant
amount of the numerical code is courtesy of Dr. Terry Haut of LANL.

Algorithm
---------
1) Set up relevant (Python) objects for computation to progress (spectral_toolbox, control, exponential_integrator)

2) Call solver method for timestepping

2a) Data is written to file at each coarse timestep

| Author: Adam G. Peddle, Terry Haut
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import logging
import time
import sys
import os
import pickle

from spectral_toolbox_1D import SpectralToolbox
import numpy as np
from numpy import fft
import cyclops_control

import matplotlib.pyplot as plt

def h_init(control, width = 2.0):
    """
    This function sets up the initial condition for the height field.

    **Returns**
    -`h_space` : The initial height field
    """

    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])
    h_space = np.exp(-width * ((x_grid-control['Lx']/2.0)**2))

    # Initial unknown vector
    U = np.vstack((np.zeros_like(h_space), np.zeros_like(h_space), h_space))

    return U

def compute_errors(U_hat_old, U_hat_new):
    """
    Call-through to the two error-computing methods.

    This method is used to analyse the iterative error and compared the current
    and previous timesteps. Both the L_inf and L_2 errors are computed.

    **Parameters**

    - `U_hat_old` : the solution at the previous timestep
    - `U_hat_new` : the solution at the current timestep

    **Returns**

    - `errors` : a list containing the L_inf and L_2 errors, respectively

    **See Also**
    `compute_L_infty_error`, `compute_L_2_error`

    """

    errors = [0., 0.]
    errors[0] = compute_L_infty_error(U_hat_old, U_hat_new)
    errors[1] = compute_L_2_error(U_hat_old, U_hat_new)
    return errors

def compute_L_2_error(U_hat_ref, U_hat_approx):
    error = abs(np.reshape(U_hat_approx[:,:] - U_hat_ref[:,:], np.prod(np.shape(U_hat_ref[:,:]))))
    error = np.sqrt(np.sum(error**2))
    norm_val = abs(np.reshape(U_hat_ref[:,:], np.prod(np.shape(U_hat_ref[:,:]))))
    norm_val = np.sqrt(np.sum(norm_val**2))
    error = error/norm_val
    return error

def compute_L_infty_error(U_hat_ref, U_hat_approx):
    N = np.shape(U_hat_ref)[-1]
    sign_mat2 = (-1)**np.arange(N)
    v1_hat = U_hat_ref[0,:]
    v2_hat = U_hat_ref[1,:]
    h_hat = U_hat_ref[2,:]
    # compute spatial solution
    v1_space = N * sign_mat2 * fft.ifft(v1_hat)
    v2_space = N * sign_mat2 * fft.ifft(v2_hat)
    h_space = N * sign_mat2 * fft.ifft(h_hat)
    # compute error in Fourier
    v1_hat_err = U_hat_approx[0,:] - U_hat_ref[0,:]
    v2_hat_err = U_hat_approx[1,:] - U_hat_ref[1,:]
    h_hat_err = U_hat_approx[2,:] - U_hat_ref[2,:]
    # compute error in space
    v1_space_err = N * sign_mat2 * fft.ifft(v1_hat_err)
    v1_space_err = max(abs(v1_space_err))/max(abs(v1_space))
    v2_space_err = N * sign_mat2 * fft.ifft(v2_hat_err)
    v2_space_err = max(abs(v2_space_err))/max(abs(v2_space))
    h_space_err = N * sign_mat2 * fft.ifft(h_hat_err)
    h_space_err = max(abs(h_space_err))/max(abs(h_space))
    error = max(v1_space_err,v2_space_err,h_space_err)
    return error


