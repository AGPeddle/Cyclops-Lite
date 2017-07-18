#!/usr/bin/env/ python3
"""
This module provides a dictionary through which all computational and physical
parameters are organised and accessed. All have semi-reasonable default values
and may be chosen by the user via the command-line interface.

Functions
---------

- `make_control` : Creates a control object with the default values
- `setup_control` : Updates the default control object with user selections

Parameters
----------

- `Nx` : Number of gridpoints along one direction. Domain is Nx X Nx square <int>
- `Nt` : Number of coarse timesteps for APinT computation <int>
- `coarse_timestep` : Coarse timestep length <float>
- `fine_timestep` : Fine timestep length <float>
- `Lx` : Side length of the square domain <float>
- `conv_tol` : The convergence criterion for iterative error <float>
- `HMM_T0` : The length of the averaging window (absolute) <float>
- `mu` : Hyperviscosity coefficient <float>
- `outFileStem` : Optional stem for output filenames <str>
- `f_naught` : Coriolis parameters <float>
- `H_naught` : Mean water depth <float>
- `gravity` : Gravitational acceleration, g <float>
- `force_gain` : The gain term for stochastic forcing <float>
- `epsilon` : Scale separation <float>
- `working_dir` : The working directory <str>
- `ensemble_size`: The number of ensemble members for data assimilation <int>
- `assim_cycle_length` : The number of coarse timesteps between EnKF update steps <int>
- `meas_spacing` : The spacing of spatial measurement points for EnKF, such that
                   data is known at 0, meas_spacing, 2*meas_spacing, ...
- `N_cycles` : The number of prediction/update cycles to be performed.

| Author: Adam G. Peddle
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import sys
import getopt
import numpy as np

def make_control():
    """
    Initialises the control object to the default values. New defaults
    should be placed here.

    **Returns**

    - `control` : Default control object
    """

    control = dict()

    # General Parameters
    control['Nx'] = 64  # Number of grid points. Domain is square.
    control['Lx'] = 2.0*np.pi  # Side length of square domain
    control['coarse_timestep'] = 0.1  # Coarse timestep
    control['fine_timestep'] = 0.1/500  # Coarse timestep
    control['solver'] = 'fine_propagator'
    control['outFileStem'] = 'test'  # Stem for creation of output files
    control['working_dir'] = None

    # Physical Parameters
    control['epsilon'] = 1.0  # Scale separation
    control['deformation_radius'] = 1.  # Rossby deformation radius (must be O(1))
    control['mu'] = 1.0e-4  # Hyperviscosity parameter

    # APinT Parameters
    control['Nt'] = 64  # Number of coarse timesteps taken
    control['conv_tol'] = 01.0e-6  # Tolerance for iterative convergence
    control['HMM_T0'] = None  # Used in wave averaging kernel

    # Data Assimilation Parameters
    control['ensemble_size'] = 32  # Number of ensemble members
    control['assim_cycle_length'] = 25  # Number of coarse timesteps between updates
    control['meas_spacing'] = 7  # Spatial spacing of measured data
    control['N_cycles'] = 6  # Number of total predict/update EnKF cycles
    control['force_gain'] = 0.0  # Gain for stochastic forcing

    return control

def setup_control(invals):
    """
    Creates and updates the default control object with user selections.
    Input should come via stdin and relies on sys.argv for parsing.

    **Parameters**

    - `invals` : command-line input values in Unix-style

    **Returns**

    - `control` : Default control object
    """

    control = make_control()

    opts, args = getopt.gnu_getopt(invals, '', ['Nx=', 'Nt=', 'coarse_timestep=', 'fine_timestep=', 'HMM_T0=', 'final_time=', 'conv_tol=', 'outFileStem=', 'delta=', 'epsilon=', 'force_gain=', 'solver=', 'ensemble_size=', 'assim_cycle_length=', 'meas_spacing=', 'N_cycles=', 'working_dir='])
    for o, a in opts:
        if o in ("--Nx"):
            control['Nx'] = int(a)
        elif o in ("--Nt"):
            control['Nt'] = int(a)
        elif o in ("--coarse_timestep"):
            control['coarse_timestep'] = float(a)
        elif o in ("--fine_timestep"):
            control['fine_timestep'] = float(a)
        elif o in ("--HMM_T0"):
            control['HMM_T0'] = float(a)
        elif o in ("--final_time"):
            control['final_time'] = float(a)
        elif o in ("--outFileStem"):
            control['outFileStem'] = a
        elif o in ("--conv_tol"):
            control['conv_tol'] = float(a)
        elif o in ("--epsilon"):
            control['epsilon'] = float(a)
        elif o in ("--force_gain"):
            control['force_gain'] = float(a)
        elif o in ("--solver"):
            control['solver'] = a
        elif o in ("--ensemble_size"):
            control['ensemble_size'] = int(a)
        elif o in ("--assim_cycle_length"):
            control['assim_cycle_length'] = int(a)
        elif o in ("--meas_spacing"):
            control['meas_spacing'] = int(a)
        elif o in ("--N_cycles"):
            control['N_cycles'] = int(a)
        elif o in ("--working_dir"):
            control['working_dir'] = a

    return control


if __name__ == "__main__":
    control = setup_control(sys.argv[1:])
    print(control)
