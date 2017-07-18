#!/usr/bin/env/ python3
"""
Library of miscellaneous helper functions for Cyclops init/admin tasks.


Functions
---------

- `read_ICs` : Read initial conditions from the Polvani experiments
- `geopotential_transform` : Transform the height field from (u,v,h) to skew-Hermitian (u,v,phi)
- `inv_geopotential_transform` : Transform the height field from skew-Hermitian (u,v,phi) to (u,v,h)
- `compute_L_2_error` : Compute the L_2 error between two vectors
- `compute_L_infty_error` : Compute the L_infty (sup-norm) error between two vectors
- `h_init` : Generate an initially stationary Gaussian height field for testing

| Authors: Adam G. Peddle
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import time
import sys
import os
import pickle

from spectral_toolbox_1D import SpectralToolbox
import numpy as np
from numpy import fft

def h_init(control, width = 2.0):
    """
    This function sets up the initial Gaussian condition for the height field.

    **Parameters**

    - `control` : control object
    - `width` : does the same thing as variance, without normalisation

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
    """
    Computes and returns the L_2 error.

    The L_2 error is defined as:

    .. math:: L_{2} = \\frac{\\sqrt{\\sum e^{2}}}{\\sqrt{\\sum u_{ref}^{2}}}

    where e is the absolute error.

    The errors are computed in Fourier space but returned in real space. The returned error
    is the L_2 error of all three variables together.

    The reference solution will generally be the solution at the previous iteration, which
    is used for measuring convergence.

    **Parameters**

    - `U_hat_ref` : the solution at the previous timestep (or a reference solution)
    - `U_hat_approx` : the solution at the current timestep
    - `st` : spectral toolbox object

    **Returns**

    - `error` : The computed L_2 error
    """

    error = abs(np.reshape(U_hat_approx[:,:] - U_hat_ref[:,:], np.prod(np.shape(U_hat_ref[:,:]))))
    error = np.sqrt(np.sum(error**2))
    norm_val = abs(np.reshape(U_hat_ref[:,:], np.prod(np.shape(U_hat_ref[:,:]))))
    norm_val = np.sqrt(np.sum(norm_val**2))
    error = error/norm_val
    return error

def compute_L_infty_error(U_hat_ref, U_hat_approx):
    """
    Compute the L_infty error at a given timestep.

    The L_infty error is defined as:

    .. math:: L_{\\infty} = \\max\\left|\\frac{U_{new}-U_{old}}{U_{old}}\\right|

    The errors are computed in Fourier space but returned in real space. The returned error
    is the greatest of the errors computed for both velocities and the height.

    The reference solution will generally be the solution at the previous iteration, which
    is used for measuring convergence.

    **Parameters**

    - `U_hat_ref` : the solution at the previous timestep (or a reference solution)
    - `U_hat_approx` : the solution at the current timestep
    - `st` : spectral toolbox object

    **Returns**

    - `error` : The computed L_infty error
    """

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


