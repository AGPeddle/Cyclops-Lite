import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import cyclops_control
import RSWE_direct
from sklearn.gaussian_process import GaussianProcessRegressor

station_spacing = 12

class Noiser:
    """

    """
    def __init__(self, mean = 0, var = 0.05):
        self.mean = mean
        self.var = var

    def apply(self, data_in):
        N = data_in.shape
        s = np.random.normal(self.mean, self.var, N)

        return data_in + s


def h_init(control):
    """
    This function sets up the initial condition for the height field.

    **Returns**
    -`h_space` : The initial height field
    """

    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])
    h_space = np.exp(-2.0 * ((x_grid-control['Lx']/3.0)**2)) + 1.5*np.exp(-3.0 * ((x_grid-2.0*control['Lx']/3.0)**2))
    return h_space


def main(control):
    """

    """
    # Hardcoded for now:
    control['Nt'] = 5

    # Set up noiser object
    noiser = Noiser()

    # Set up initial (truth) field
    truth = h_init(control)

    # Propagate it through Nts
    truth = RSWE_direct.solve(control, truth)

    # Noisify the data
    truth = noiser.apply(truth)
    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])
    """
    for i in range(control['Nt'] + 1):
        plt.plot(x_grid, truth[i,2,:], label = "{}".format(i))
    plt.legend()
    plt.show()
    """

    # Create Gaussian process
    gpr = GaussianProcessRegressor()
    x_grid = x_grid.reshape(-1,1)
    x_learn = x_grid[0::station_spacing]
    y_learn = truth[0,2,0::station_spacing]
    gpr.fit(x_learn, y_learn)

    # Produce initial conditions
    for i in range(5):
        plt.plot(x_grid, gpr.sample_y(x_grid, random_state=i), label = "{}".format(i))
    plt.legend()
    plt.show()

    # Pickle observations and ICs (securely?)
    assim_data = dict()
    assim_data['x'] = x_learn
    assim_data['t'] = np.arange(0, (control['Nt']+1)*control['timestep'], control['timestep'])
    assim_data['u'] = truth[:,:,0::station_spacing]

    with open('assim_data.dat' 'wb') as f:
        pickle.dump(assim_data, f)


if __name__ == "__main__":
    control_in = cyclops_control.setup_control(sys.argv[1:])
    main(control_in)
