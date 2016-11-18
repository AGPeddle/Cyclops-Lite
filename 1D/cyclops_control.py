import sys
import getopt
import numpy as np

def make_control():
    control = dict()

    control['Nx'] = 64  # Number of grid points. Domain is square.
    control['delta'] = 500  # Number of fine timesteps to take per coarse step
    control['timestep'] = 0.1  # Coarse timestep
    control['final_time'] = 1.0  # Final time for computation
    control['Lx'] = 2.0*np.pi  # Side length of square domain
    control['conv_tol'] = 01.0e-4  # Tolerance for iterative convergence
    control['kernel'] = True

    control['epses'] = []
    control['HMM_T0'] = 0.5  # Used in wave averaging kernel

    control['deformation_radius'] = 1.  # Rossby deformation radius (must be O(1))
    control['epsilon'] = 1.0  # Epsilon (Rossby) can be 0.01, 0.1, or 1.0
    control['mu'] = 1.0e-4  # Hyperviscosity parameter

    control['outFileStem'] = 'test'  # Stem for creation of output files
    control['rank'] = 0  # Rank number for parallel computation. Set to 0 for serial safety.
    control['gauss_width'] = 2.0

    return control

def setup_control(invals):

    control = make_control()

    opts, args = getopt.gnu_getopt(invals, '', ['Nx=', 'timestep=', 'final_time=', 'outFileStem=', 'delta=', 'epsilon=', 'eps1=', 'eps2=', 'eps3=', 'eps4=', 'eps5=', 'eps6=', 'eps7=', 'eps8=', 'nokernel'])
    for o, a in opts:
        if o in ("--Nx"):
            control['Nx'] = int(a)
        elif o in ("--timestep"):
            control['timestep'] = float(a)
        elif o in ("--final_time"):
            control['final_time'] = float(a)
        elif o in ("--outFileStem"):
            control['outFileStem'] = a
        elif o in ("--delta"):
            control['delta'] = int(a)
        elif o in ("--epsilon"):
            control['epsilon'] = float(a)
        elif o in ("--eps1"):
            control['epses'].append(float(a))
        elif o in ("--eps2"):
            control['epses'].append(float(a))
        elif o in ("--eps3"):
            control['epses'].append(float(a))
        elif o in ("--eps4"):
            control['epses'].append(float(a))
        elif o in ("--eps5"):
            control['epses'].append(float(a))
        elif o in ("--eps6"):
            control['epses'].append(float(a))
        elif o in ("--eps7"):
            control['epses'].append(float(a))
        elif o in ("--eps8"):
            control['epses'].append(float(a))
        elif o in ("--nokernel"):
            control['kernel'] = False

    return control
if __name__ == "__main__":
    control = setup_control(sys.argv[1:])
    print(control)
