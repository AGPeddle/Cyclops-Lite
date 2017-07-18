#!/usr/bin/env/ python3
"""
An implementation of the Ensemble Kalman filter is provided for sequential data
assimilation. Use is through the Ensemble_Kalman class.

The EnKF aproximates the probability distribution of the solution through a
finite ensemble of members, (see also Markov Chain Monte Carlo). This estimate
is updated at each prediction step in a Bayesian fashion, with the predicted
ensemble forming the prior. Upon incorporating the measurement data, we have a
posterior distribution.

Classes
-------
- `Ensemble_Kalman` : Implements the filter.

| Author: Adam G. Peddle
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cyclops_control
import RSWE_direct
import time

class Ensemble_Kalman:
    """
    This class implements the update step of the ensemble Kalman filter. Once
    initialised, the point of access for the user is through the apply method.

    **Attributes**

    - `Nx` : The spatial domain size
    - `N` : the number of ensemble members
    - `H` : the measurement operator such that

    .. math :: d = H\psi

    where :math:`d` is the vector of observations, :math:`H` is the measurement
    operator, and :math:`\psi` is the full solution vector.

    **Methods**

    - `_ensemble_perturbation` : Computes the ensemble perturbation
    - `_perturb_observations` : Computes the perturbed observations and
                                ensemble of perturbations
    - `apply`: Applies the EnKF to a predicted solution for some measurement

    **Parameters**

    - `x_meas` : The indices where measurements are known on the solution grid
    - `Nx` : The number of gridpoints
    - `ensemble_size` : The number of ensemble members

    **See Also**

    - G. Evensen, The Ensemble Kalman Filter: theoretical formulation and practical implementation. 2003. Ocean Dynamics. DOI:10.1007/s10236-003-0036-9.
    """

    def __init__(self, x_meas, Nx, ensemble_size):
        self.Nx = Nx
        self.N = ensemble_size

        self.H = np.zeros((x_meas.shape[0], self.Nx))
        self.H[np.arange(x_meas.shape[0]), x_meas] = 1

        np.random.seed(0)

    def _ensemble_perturbation(self, A):
        """
        Computes the ensemble perturbation matrix of A, i.e.

        .. math :: A' = A - \\overline{A}

        **Parameters**

        - `A` : Array holding the ensemble members. Is of size :math:`n\timesN`
                where n is the size of the model state vector and N is the
                number of ensemble members.

        **Returns**

        - The ensemble perturbation matrix.
        """

        X = np.identity(self.N) - (1/self.N)*np.ones((self.N, self.N))
        return np.dot(A, X)

    def _perturb_observations(self, d):
        """
        Creates a perturbed observation matrix from a vector of observations.

        **Parameters**

        - `d` : The vector of measurements (length m = number of measurements)

        **Returns**

        - `D` : The perturbed observation matrix
                (size m x N, N is num ensemble members)
        - `ens_perts` : The perturbations employed (size m x N)

        **Notes**

        The perturbations should have mean zero, which is hardcoded. Variance
        can in theory be varied, but this is not implemented here.

        """

        mean = 0
        var = 0.10

        D = np.tile(d, (1,self.N))
        ens_perts = np.random.normal(mean, var, D.shape)

        return D, ens_perts

    def apply(self, A, d):
        """
        Applies the EnKF, updating the predicted ensemble matrix with the
        measured data. The result is the posterior distribution, i.e. the best
        estimate for the state of the system.

        Let n be the size of the model state vector, N be the number of
        ensemble members, and m be the number of measurements.

        **Parameters**

        - `A` : the prior distribution (real, size n x N)
        - `d` : the measurement vector (real, size m)

        **Returns**

        - The posterior distribution after data has been assimilated. This
          matrix is suitable for continued use in MCMC solvers.

        """

        A_p = self._ensemble_perturbation(A)
        P_e = self._ensemble_perturbation(A_p)

        D, ens_perts = self._perturb_observations(d)
        R_e = np.dot(ens_perts, ens_perts.T)/(self.N-1)
        D_p = D - np.dot(self.H, A)

        A2 = np.dot(A_p, A_p.T)
        X = np.dot(self.H, A2)
        X = np.dot(X, self.H.T) + np.dot(ens_perts, ens_perts.T)

        X = np.linalg.inv(X)  # May need Moore-Penrose pseudo-inverse

        RHS = np.dot(A2, self.H.T)
        RHS = np.dot(RHS, X)
        RHS = np.dot(RHS, D_p)

        return A + RHS

