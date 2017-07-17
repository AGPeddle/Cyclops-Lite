#!/usr/bin/env/ python3
"""
"""

from spectral_toolbox_1D import SpectralToolbox
import numpy as np
import matplotlib.pyplot as plt

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

        - `U_hat` : the solution at the current timestep

        **Returns**

        - `u_hat_new` : the solution at the next timestep

        """

        U_hat_new = np.zeros(np.shape(U_hat), dtype = 'complex')
        U_hat_old = U_hat.copy()

        t = 0
        while t < control['final_time']:
            "limit fine timestep size to avoid overshooting the last timestep"
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

        - `U_hat` : the solution at the current timestep

        **Returns**

        - `u_hat_new` : the solution at the next timestep

        """
        U_hat_old = U_hat.copy()
        t = 0
        while t < control['final_time']:
            "limit fine timestep size to avoid overshooting the last timestep"
            dt = min(control['coarse_timestep'], control['final_time']-t)

            U_hat_new = strang_splitting(U_hat_old, dt, control, expInt, st, dissipative_exponential, compute_average_force)
            U_hat_new = expInt.call_(U_hat_new, dt)

            U_hat_old = U_hat_new
            t += dt

        return U_hat_new


class ExponentialIntegrator:
    """
    Implements the exponential integrator for the Rotating Shallow Water Equations.

    This class implements the exponential integrator objects, which precompute and store
    the eigenvalues and eigenvectors of the linear operator in Fourier space and apply
    the matrix exponential for a given timestep when invoked. See module header for
    explanation of computation.

    **Attributes**

    - `deriv_factor` : factor of 2*pi/L due to spectral differentiation
    - `_Nx` : Number of grid points, such that domain is Nx X Nx
    - `eVals` : Vector of eigenvalues, ordered alpha = 0, -1, 1
    - `eBasis` : Corresponding orthonormalised matrix of eigenvectors, arranged in columns
    - `_Froude` : Froude number
    - `_eps` : Epsilon, used in QG formulation
    - `_F` : Rossby deformation radius

    **Methods**

    - `create_eigenvalues` : Set up the analytically-computed eigenvalues for L
    - `create_eigenbasis` : Set up the analytically-computed eigenbasis for L
    - `call_` : Invoke the matrix exponential

    **Parameters**

    - `control` : Control object containing the relevant parameters/constants for initialisation

    """

    def __init__(self, control):
        # Factor arising from spectral derivative on domain of size L
        self.deriv_factor = 2.0*np.pi/control['Lx']

        self._Nx = control['Nx']
        self.eVals  = np.zeros((3,self._Nx),   dtype = np.float64)
        self.eBasis = np.zeros((3,3,self._Nx), dtype = complex)

        ctr_shift = self._Nx//2  # Python arrays are numbered from 0 to N-1, spectrum goes from -N/2 to N/2-1

        self._Froude = np.sqrt(control['deformation_radius'])*control['epsilon']
        self._eps = control['epsilon']
        self._F = control['deformation_radius']

        for k in range(-self._Nx//2, self._Nx//2):
            self.eVals[:,k + ctr_shift]    = self.create_eigenvalues(k)
            self.eBasis[:,:,k + ctr_shift] = self.create_eigenbasis(k)

    def create_eigenvalues(self, k):
        """

        Creates the eigenvalues required for the creation of the eigenbasis (of the linear operator).
        Note that these have been found analytically and so an arbitrary L is not currently supported.

        The eigenvalues returned are those for the QG formulation of the RSWE.

        **Parameters**
        -`k1, k2` : The wavenumber pair for which the eigenvalue is desired

        **Returns**
        -`eVals` : The computed eigenvalues

        """

        eVals = np.zeros(3, dtype = np.float64)
        eVals[0] = 0.
        eVals[1] = -np.sqrt(1. + (k*k)/self._F)
        eVals[2] = np.sqrt(1. + (k*k)/self._F)

        return eVals

    def create_eigenbasis(self, k):
        """
        Create the eigenbasis of the linear operator. As with the eigenvalues, it is an implementation of
        an analytical solution.

        **Parameters**
        -`k1, k2` : The wavenumber pair for which the eigenvalue is desired

        **Returns**
        -`A` : The computed eigenvector array

        """

        A = np.zeros((3,3), dtype = complex)
        if k != 0:
            f = np.sqrt(self._F)
            kappa = k*k
            omega = 1j*f*np.sqrt(1 + kappa/(f**2))

            # Eigenvectors in columns corresponding to slow mode and
            # fast waves travelling in either direction
            A = np.array([[0.0,     1j*omega/k, -1j*omega/k],
                          [1j*k/f,  -1j*f/k,    -1j*f/k],
                          [1.,       1.,        1.]], dtype = complex)

            A[:,0] = A[:,0]/np.sqrt(1 + 1/f**2 * kappa)
            A[:,1] = A[:,1]/np.sqrt(2. + 2.*f**2/float(kappa))
            A[:,2] = A[:,2]/np.sqrt(2. + 2.*f**2/float(kappa))

        else:  # Special case for k1 = k2 = 0
            A = np.array([[0., -1j/np.sqrt(2.), 1j/np.sqrt(2.)],\
                          [0., 1./np.sqrt(2.),  1./np.sqrt(2.)],\
                          [1., 0.,              0.]],dtype = complex)

        return A

    def call_(self,U_hat,t):
        """
        Call to invoke the exponential integrator on a given initial solution over some time.

        Propagates the solution to the linear problem for some time, t, based on the initial
        condition, U. U is, in general, the value at the beginning of a timestep and t is the
        timestep, although this need not be the case.

        **Parameters**

        - `U_hat` : the Fourier modes of the unknown solution, ordered as below
        - `t` : the time at which the solution is desired (or timestep to be used)

        **Returns**

        - `U_hat_sp` : The computed solution, propagated by time t

        **Notes**

        1) Works in -t direction as well.
        2) Input order: v1[:,:] = U_hat[0,:,:], v2[:,:] = U_hat[1,:,:], h[:,:] = U_hat[2,:,:]

        """

        # Separate unknowns to allow vector multiply for speed
        v1_hat = U_hat[0,:]
        v2_hat = U_hat[1,:]
        h_hat = U_hat[2,:]

        # First eigenvector for L(ik), eigenvalue omega=0
        rk00 = self.eBasis[0,0,:]
        rk10 = self.eBasis[1,0,:]
        rk20 = self.eBasis[2,0,:]

        # Second eigenvector for L(ik),
        # eigenvalue omega = -i*sqrt(F+k^2)
        rk01 = self.eBasis[0,1,:]
        rk11 = self.eBasis[1,1,:]
        rk21 = self.eBasis[2,1,:]

        # Third eigenvector for L(ik),
        # eigenvalue omega = I*sqrt(F+k^2)
        rk02 = self.eBasis[0,2,:]
        rk12 = self.eBasis[1,2,:]
        rk22 = self.eBasis[2,2,:]

        # Convert to eigenvector basis
        U_hat_sp = np.zeros((3,self._Nx), dtype='complex')
        v1_hat_sp = np.conj(rk00) * v1_hat + np.conj(rk10) * v2_hat + np.conj(rk20) * h_hat
        v2_hat_sp = np.conj(rk01) * v1_hat + np.conj(rk11) * v2_hat + np.conj(rk21) * h_hat
        h_hat_sp = np.conj(rk02) * v1_hat + np.conj(rk12) * v2_hat + np.conj(rk22) * h_hat

        # Apply exp(-t*L)
        omega0 = self.eVals[0,:]; omega0 = np.exp(-1j*omega0*t/self._eps)
        omega1 = self.eVals[1,:]; omega1 = np.exp(-1j*omega1*t/self._eps)
        omega2 = self.eVals[2,:]; omega2 = np.exp(-1j*omega2*t/self._eps)

        U_hat_sp[0,:] = omega0 * rk00 * v1_hat_sp +  omega1 * rk01 * v2_hat_sp + \
                        omega2 * rk02 * h_hat_sp
        U_hat_sp[1,:] = omega0 * rk10 * v1_hat_sp +  omega1 * rk11 * v2_hat_sp + \
                        omega2 * rk12 * h_hat_sp
        U_hat_sp[2,:] = omega0 * rk20 * v1_hat_sp +  omega1 * rk21 * v2_hat_sp + \
                        omega2 * rk22 * h_hat_sp

        return U_hat_sp

########## BEGIN SOLVER (SUB) ROUTINES ##########

def dissipative_exponential(control, expInt, U_hat, t):
    """
    Implements dissipation through a 4th-order hyperviscosity operator
    and matrix exponential.

    **Parameters**

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
    -`U_hat` : the components of the unknown vector in Fourier space, ordered u, v, h
    -`linear_operator` : not used here. Required to have identical calling to compute_average_force

    **Returns**
    -`U_NL_hat` : the result of the multiplication in Fourier space

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
    - `linear_operator` : the linear operator being used, to be
      passed to the nonlinear computation

    **Returns**

    - `U_hat_averaged` : The predicted averaged solution at the next timestep

    **Notes**

    The smooth kernel is chosen so that the length of the time window over which the averaging is
    performed is as small as possible and the error from the trapezoidal rule is negligible.

    *See Also**
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
