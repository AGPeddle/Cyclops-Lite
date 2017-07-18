#!/usr/bin/env/ python3
"""
This module implements the solution to the exponential integrator for the linear operator
of the rotating shallow water equations as used by pypint.

Consider the RSWE, neglecting the nonlinear terms. We may write this as:

.. math :: \\frac{\\partial u}{\\partial t} = Lu

We may solve this via an integrating factor method, yielding:

.. math:: u_{t} = e^{tL}u_{0}

to which we may freely apply various choices of timestepping. The exponential integrator
then requires the efficient computation of the matrix exponential. We write the matrix
exponential in the form:

.. math:: e^{tL} = r_{k}^{\\alpha} e^{t\\omega_{k}^{\\alpha}} (r_{k}^{\\alpha})^{-1}

where :math:`r_{k}^{\\alpha}` is a matrix containing the eigenvectors of the linear
operator (for a corresponding wavenumber, k, as we are working in Fourier space) and
:math:`\\omega_{k}^{\\alpha}` is a vector containing the associated eigenvalues of
the linear operator, such that:

.. math:: \\omega_{k}^{\\alpha} = \\alpha\\sqrt{1+F^{-1}k^{2}}

with :math:`\\alpha = -1, 0, 1` and where F is the deformation radius. This
reduces the problem to an easier problem of finding the eigenvalues/eigenvectors
and applying these. These are pre-computed for speed as they will not change
over the course of the computation for constant gravity, rotation, and water depth.

Classes
-------

-`ExponentialIntegrator` : Implements the exponential integrator for the RSWE

Notes
-----

We write the linear operator, L, as:

.. math:: L = \\left[\\begin{array}{ccc}
                0 & -1 & F^{-1}\\partial_{x} \\\\
               1 & 0 & 0 \\\\
               F^{-1}\\partial_{x} & 0 & 0 \\end{array}\\right]

which becomes, in Fourier space:

.. math:: L = \\left[\\begin{array}{ccc}
                0 & -1 & 2\\pi i F^{-1} k_{1}/L \\\\
               1 & 0 & 0 \\\\
               2\\pi i F_{-1} k_{1}/L & 0 & 0 \\end{array}\\right]

See Also
--------

numpy

| Authors: Terry Haut, Adam G. Peddle
| Contact: ap553@exeter.ac.uk
| Version: 2.0
"""

import numpy as np

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


