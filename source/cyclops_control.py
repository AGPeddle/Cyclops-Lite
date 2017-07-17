import sys
import getopt
import numpy as np

def make_control():
    control = dict()

    control['Nx'] = 64  # Number of grid points. Domain is square.
    control['Nt'] = 64  # Number of coarse timesteps taken
    control['coarse_timestep'] = 0.1  # Coarse timestep
    control['fine_timestep'] = 0.1/500  # Coarse timestep
    control['final_time'] = 1.0  # Final time for computation
    control['Lx'] = 2.0*np.pi  # Side length of square domain
    control['conv_tol'] = 01.0e-6  # Tolerance for iterative convergence
    control['solver'] = 'fine_propagator'

    control['epsilon'] = 1.0
    control['HMM_T0'] = None  # Used in wave averaging kernel
    control['force_gain'] = 0.0  # Gain for stochastic forcing

    control['deformation_radius'] = 1.  # Rossby deformation radius (must be O(1))
    control['epsilon'] = 1.0  # Epsilon (Rossby) can be 0.01, 0.1, or 1.0
    control['mu'] = 1.0e-4  # Hyperviscosity parameter

    control['outFileStem'] = 'test'  # Stem for creation of output files
    control['working_dir'] = None
    control['gauss_width'] = 2.0

    control['ensemble_size'] = 32
    control['assim_cycle_length'] = 25
    control['meas_spacing'] = 7
    control['N_cycles'] = 6

    return control

def setup_control(invals):

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
