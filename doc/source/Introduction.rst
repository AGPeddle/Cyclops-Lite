Introduction
============
Cyclops is a program to solve the Rotating Shallow Water Equations (RSWE) using a
Parareal solver, and in particular the novel Asymptotic Parallel-in-Time (APiNT)
method. The APinT method is described by `Haut and Wingate, 2014 <http://arxiv.org/abs/1303.6615>`.

The program uses a Fourier spectral method and solves the equations on a 1-D
doubly-periodic domain with evenly-spaced gridpoints.

The purpose of this code is to act as a mathematical testbed to investigate the
mathematical basis of the APinT method, particularly the wave averaging for oscillatory-
stiff problems. It is hoped that future collaborators may easily work with and modify
the code to improve the understanding of this class of methods.

This lite version has been stripped of the parallel capabilities of the full Cyclops
implementation (i.e. it is a serial implementation of a parallel algorithm) in order
to facilitate easy use on single-core or few-core machines.

In addition to cyclops-lite, which has cyclops.py as its point of access, this
repository also contains the faro project, which is used for sequential data
assimilation via the ensemble Kalman filter. The same wave-averaging-based
coarse solver used in APinT can be used here as well. For this reason, linear
stochastic forcing is implemented in cyclops-lite.

Dependencies
------------
- Python 3
- numpy
- pickle
- pyfftw

Authors
-------
The authors of Cyclops are Adam G. Peddle (ap553@exeter.ac.uk) and Dr. Terry Haut.
