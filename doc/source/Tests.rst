Tests
=====

We currently provide two tests. These are both invocable from the command-line
interface.

These tests also provide examples of calling Cyclops.

APinT.sh
----------------

This is the first test to be run and should converge very quickly. An initially
stationary Gaussian height field is assumed on a coarse spatial grid. The test
runs the APinT parallel-in-time algorithm for the 1-D RSWE.

EnKF.sh
--------

This test is much more computationally intensive. It creates the necessary
data to simulate a data assimilation cycle and then performs this cycle, using
the coarse propagator for the prediction step and the ensemble Kalman filter
for the update step.
