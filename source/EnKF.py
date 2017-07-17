import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cyclops_control
import RSWE_direct
import time

class Ensemble_Kalman:
    def __init__(self, x_meas, Nx, ensemble_size):
        self.Nx = Nx
        self.N = ensemble_size

        self.H = np.zeros((x_meas.shape[0], self.Nx))
        self.H[np.arange(x_meas.shape[0]), x_meas] = 1

        np.random.seed(0)

    def ensemble_perturbation(self, A):
        X = np.identity(self.N) - (1/self.N)*np.ones((self.N, self.N))
        return np.dot(A, X)

    def ensemble_covariance(self, A):
        A_p = self.ensemble_perturbation(A)

        return (1/(self.N-1))*np.outer(A_p, A_p.T)

    def perturb_observations(self, d):

        mean = 0
        var = 0.10

        D = np.tile(d, (1,self.N))
        ens_perts = np.random.normal(mean, var, D.shape)

        return D, ens_perts

    def apply(self, A, d):
        A_p = self.ensemble_perturbation(A)
        P_e = self.ensemble_perturbation(A_p)

        D, ens_perts = self.perturb_observations(d)
        R_e = np.dot(ens_perts, ens_perts.T)/(self.N-1)
        D_p = D - np.dot(self.H, A)

        A2 = np.dot(A_p, A_p.T)
        X = np.dot(self.H, A2)
        X = np.dot(X, self.H.T) + np.dot(ens_perts, ens_perts.T)

        X = np.linalg.inv(X)

        RHS = np.dot(A2, self.H.T)
        RHS = np.dot(RHS, X)
        RHS = np.dot(RHS, D_p)

        return A + RHS

