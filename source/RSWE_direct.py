#!/usr/bin/env/ python3
"""
Module to handle the generic solvers for the RSWE. The point of
access is through the solve function.

Functions
---------

- `fine_propagator` -- computes the full (unaveraged) RSWE by Strang splitting
- `coarse_propagator` -- access point for averaged RSWE timestepping
- `dissipative_exponential` -- (hyper)viscosity operator
- `exp_L_exp_D` -- exponentiated sum of linear terms (advective and dissipative)
- `strang_splitting` -- generic strang splitting method
- `compute_nonlinear` -- computes the nonlinear terms for the perturbed RSWE
- `filter_kernel_exp` -- provides a normalised exponential kernel of integration
- `compute_average_force` -- computes the averaged nonlinearity (RHS)
- `solve` -- point of access for this library

| Authors: Adam G. Peddle, Terry Haut
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

from spectral_toolbox_1D import SpectralToolbox
import numpy as np

class Solvers:
    @staticmethod
    def fine_propagator(control, expInt, st, U_hat):
        """
        Implements the fine propagator used in the APinT or Parareal methods. Can handle
        any implemented methods for the linear and nonlinear operator. Calls through to
        appropriate methods.

        The full range of fine timesteps is taken in this module from the previous to the
        next coarse timestep. Only the solution corresponding to the desired coarse timestep
        is returned; all others are discarded.

        **Parameters**

        - `control` : control object
        - `expInt` : appropriate exponential integrator
        - `st` : spectral toolbox objecy
        - `U_hat` : the solution at the current timestep

        **Returns**

        - `U_hat_new` : the solution at the next timestep

        """

        U_hat_new = np.zeros(np.shape(U_hat), dtype = 'complex')
        U_hat_old = U_hat.copy()

        t = 0
        while t < control['final_time']:
            # limit fine timestep size to avoid overshooting the last timestep
            dt = min(control['fine_timestep'], control['final_time']-t)

            U_hat_new = strang_splitting(U_hat_old, dt, control, expInt, st, exp_L_exp_D, compute_nonlinear)

            U_hat_old = U_hat_new
            t += dt

        return U_hat_new

    @staticmethod
    def coarse_propagator(control, expInt, st, U_hat):
        """
        Implements the coarse propagator used in the APinT method. Can handle any implemented methods
        for the linear and nonlinear operator. Calls through to appropriate methods. Differs from the
        non_asymptotic version in its call to the average force computation.

        **Parameters**

        - `control` : control object
        - `expInt` : appropriate exponential integrator
        - `st` : spectral toolbox objecy
        - `U_hat` : the solution at the current timestep

        **Returns**

        - `U_hat_new` : the solution at the next timestep

        """

        U_hat_old = U_hat.copy()
        t = 0
        while t < control['final_time']:
            # limit fine timestep size to avoid overshooting the last timestep
            dt = min(control['coarse_timestep'], control['final_time']-t)

            U_hat_new = strang_splitting(U_hat_old, dt, control, expInt, st, dissipative_exponential, compute_average_force)
            U_hat_new = expInt.call_(U_hat_new, dt)

            U_hat_old = U_hat_new
            t += dt

        return U_hat_new

########## BEGIN SOLVER (SUB) ROUTINES ##########

def dissipative_exponential(control, expInt, U_hat, t):
    """
    Implements dissipation through a 4th-order hyperviscosity operator
    and matrix exponential.

    If (linear) stochastic forcing is being used, the white noise
    is implemented here as well.

    Caveat: the hyperviscosity is not normalised to grid spacing.

    **Parameters**

    - `control` : control object
    - `expInt` : exponential integrator for linear term
    - `U_hat` : the known solution at the current timestep
    - `t` : the timestep taken

    **Returns**

    - `U_hat_sp` : The solution at the next timestep by dissipation
    """

    U_hat_sp = np.zeros((3, control['Nx']), dtype = 'complex')
    wavenumbers_x = np.arange(-control['Nx']/2, control['Nx']/2)

    wavenumbers_forcing = np.zeros_like(wavenumbers_x)
    gain = control['force_gain']
    for i in range(1, 5):
        forcing = np.random.normal(loc=0.0, scale=1.0)
        wavenumbers_forcing[control['Nx']//2 + i] = gain*forcing
        wavenumbers_forcing[control['Nx']//2 - i] = -gain*forcing

    exp_D = np.exp(-control['mu']*t*(wavenumbers_x**4) + np.sqrt(t)*wavenumbers_forcing)

    for k in range(3):
        U_hat_sp[k,:] = exp_D*U_hat[k,:]
    return U_hat_sp

def exp_L_exp_D(control, expInt, U_hat, t):
    """
    Call-through method for applying the linear solution
    and the dissipative term. (Both via exponential integrator)

    **Parameters**

    - `control` : control object
    - `expInt` : exponential integrator for linear term
    - `U_hat` : the known solution at the current timestep
    - `t` : the timestep taken

    **Returns**

    - `U_hat_sp` : The solution propagated by L and D

    """

    U_hat_sp = expInt.call_(U_hat, t)  # exp(Lt/epsilon)
    U_hat_sp = dissipative_exponential(control, expInt, U_hat_sp, t)  # Dissipative term
    return U_hat_sp

def strang_splitting(U_hat, delta_t, control, expInt, st, linear, nonlinear):
    """
    Propagates a solution for arbitrary linear and nonlinear operators over a timestep
    delta_t by using Strang Splitting.

    **Parameters**

    - `U_hat` : The known solution at the current timestep, in Fourier space
    - `delta_t` : the timestep over which the solution is to be predicted
    - `control` : the control object
    - `expInt` : the exponential integrator (linear term)
    - `st` : spectral toolbox object
    - `linear` : the linear term. May be the same as expInt, but necessary
                 for integration with exp_L_exp_D
    - `nonlinear` : the nonlinear operator being used, i.e. with or without wave averaging

    **Returns**

    - `U_hat_new` : The computed U_hat at the next timestep

    """

    U_hat_new = linear(control, expInt, U_hat, 0.5*delta_t)
    # Computation of midpoint:
    U_NL_hat = nonlinear(U_hat_new, control, st, expInt)
    U_NL_hat1 = -delta_t*U_NL_hat

    U_NL_hat = nonlinear(U_hat_new + 0.5*U_NL_hat1, control, st, expInt)
    U_NL_hat2 = -delta_t*U_NL_hat
    U_hat_new = U_hat_new + U_NL_hat2

    U_hat_new = linear(control, expInt, U_hat_new, 0.5*delta_t)
    return U_hat_new

def compute_nonlinear(U_hat, control, st, expInt = None):
    """
    Function to compute the nonlinear terms of the RSWE. Calls through to some
    spectral toolbox methods to implement derivatives and nonlinear multiplication.

    This function implements the simple solution to the problem, with none of the wave averaging.

    **Parameters**

    - `U_hat` : the components of the unknown vector in Fourier space, ordered u, v, h
    - `control` : control object
    - `st` : spectral toolbox object
    - `expInt` : exponential integrator object

    **Returns**

    - `U_NL_hat` : the result of the multiplication in Fourier space

    **See Also**

    compute_average_force

    """
    N = np.shape(U_hat)[1]
    # Note that the size of U_hat_NL is double that of U_hat, for aliasing control
    U_NL_hat = np.zeros((3,N), dtype='complex')
    v1_hat = U_hat[0,:]
    v2_hat = U_hat[1,:]
    h_hat = U_hat[2,:]

    # Compute Fourier coefficients of v1 * v1_x1
    U_NL_hat1 = st.multiply_nonlinear(st.calc_derivative(v1_hat), v1_hat)
    U_NL_hat[0,:] = U_NL_hat1

    # Compute Fourier coefficients of  v1 * v2_x1
    U_NL_hat1 = st.multiply_nonlinear(st.calc_derivative(v2_hat), v1_hat)
    U_NL_hat[1,:] = U_NL_hat1

    # Compute Fourier coefficients of (h*v1)_x1
    U_NL_hat1 = st.multiply_nonlinear(h_hat, v1_hat)
    U_NL_hat1 = st.calc_derivative(U_NL_hat1)
    U_NL_hat[2,:] = U_NL_hat1

    return U_NL_hat

########## END SOLVER (SUB) ROUTINES ##########

########## BEGIN WAVE AVERAGING ROUTINES ##########

def filter_kernel_exp(M, s):
    """
    Smooth integration kernel.

    This kernel is used for the integration over the fast waves. It is
    formulated as:

    .. math:: \\rho(s) \\approx \\exp(-50*(s-0.5)^{2})

    and is normalised to have a total integral of unity. This method is
    used for the wave averaging, which is performed by `self.compute_average_force`.

    **Parameters**

    - `M` : The interval over which the average is to be computed
    - `s` : The sample in the interval

    **Returns**

    The computed kernel value at s.

    """

    points = np.arange(1, M)/float(M)
    norm = (1.0/float(M))*sum(np.exp(-50*(points - 0.5)**2))
    return np.exp(-50*(s-0.5)**2)/norm

def compute_average_force(U_hat, control, st, expInt):
    """
    This method computed the wave-averaged solution for use with the APinT
    coarse timestepping.

    The equation solved by this method is a modified equation for a slowly-varying
    solution (see module header, above). The equation solved is:

    .. math:: \\bar{N}(\\bar{u}) = \\sum \\limits_{m=0}^{M-1} \\rho(s/T_{0})e^{sL}N(e^{-sL}\\bar{u}(t))

    where :math:`\\rho` is the smoothing kernel (`filter_kernel`).

    **Parameters**

    - `U_hat` : the known solution at the current timestep
    - `control` : control object
    - `st` : spectral toolbox object
    - `expInt` : exponential integrator for linear term

    **Returns**

    - `U_hat_averaged` : The predicted averaged solution at the next timestep

    **Notes**

    The smooth kernel is chosen so that the length of the time window over which the averaging is
    performed is as small as possible and the error from the trapezoidal rule is as small as possible.

    **See Also**

    `filter_kernel`

    """

    T0 = control['HMM_T0']
    M = control['HMM_M_bar']
    filter_kernel = filter_kernel_exp

    U_hat_NL_averaged = np.zeros(np.shape(U_hat), dtype = 'complex')

    for m in np.arange(1,M):
        tm = T0*m/float(M)
        Km = filter_kernel(M, m/float(M))
        U_hat_RHS = expInt.call_(U_hat, tm)
        U_hat_RHS = compute_nonlinear(U_hat_RHS, control, st)
        U_hat_RHS = expInt.call_(U_hat_RHS, -tm)

        U_hat_NL_averaged += Km*U_hat_RHS
    return U_hat_NL_averaged/float(M)

########## END WAVE AVERAGING ROUTINES ##########

def solve(control, expInt, st, u_init, solver = None, realspace = True):
    """
    Main point of access for the RSWE solver library. Handles either real-space
    input or Fourier-space input via the ivnert_fft flag.

    **Parameters**

    - `solver_name` : string 'fine_propagator' or 'coarse_propagator' to
                      toggle solvers
    - `control` : control object
    - `st` : spectral toolbox objecy
    - `expInt` : appropriate exponential integrator (contructed externally for speed)
    - `u_init` : the solution at the current timestep, i.e. initial condition
    - `solver` : string override for control['solver']
    - `invert_fft` : flag to switch on FFT-ing. True if u_init is in realspace

    **Returns**

    - `out_sols` : the solution at the right-side of the time interval

    **Notes**

    u_init must be a numpy array of size (3, N_x, N_x) with
    solution fields along the first rank in the order (u, v, h).
    """

    if solver is None: solver = control['solver']

    if realspace:  # i.e. initial condition is in realspace
        out_sols = np.zeros((3, control['Nx']), dtype = 'complex')

        for k in range(3):  # Transform to Fourier space
            out_sols[k, :] = st.forward_fft(u_init[k, :])

        # Solver kernel
        out_sols = getattr(Solvers, solver)(control, expInt, st, out_sols)

        for k in range(3):  # Return to realspace
            out_sols[k, :] = st.inverse_fft(out_sols[k, :])

        return np.real(out_sols)

    else:  # Initial condition is already in Fourier space
        # Solver kernel
        out_sols = getattr(Solvers, solver)(control, expInt, st, u_init)

        return out_sols
