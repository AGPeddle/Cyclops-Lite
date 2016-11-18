import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import cyclops_control
import RSWE_direct
from sklearn.gaussian_process import GaussianProcessRegressor

station_spacing = 14
n_inits = 32

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
    control['solver'] = "fine_propagator"

    # Set up noiser object
    noiser = Noiser()

    # Set up initial (truth) field
    truth = np.zeros((3, control['Nx']))
    truth[2,:] = h_init(control)

    # Propagate it through Nts
    truth = RSWE_direct.solve(control, truth)
    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])

    """
    for i in range(control['Nt'] + 1):
        plt.plot(x_grid, truth[i,2,:], label = "Timestep{}".format(i))
    plt.legend()
    plt.xlabel("x-coordinate")
    plt.ylabel("Height Field")
    plt.title("Flow Truth, epsilon = {}".format(control['epsilon']))
    plt.show()
    """

    # Noisify the data
    noisy_truth = noiser.apply(truth)
    x_sample = np.append(np.arange(0,x_grid.shape[0],station_spacing),-1)

    """
    colours = ['r','g','b','c','k','y']
    for i in range(control['Nt'] + 1):
        plt.plot(x_grid, truth[i,2,:], '-', color = colours[i],  label = "Timestep {}".format(i))
        plt.plot(x_grid[x_sample], noisy_truth[i,2,x_sample], 'o', color = colours[i])
    plt.legend()
    plt.xlabel("x-coordinate")
    plt.ylabel("Height Field")
    plt.title("Flow Truth with Gaussian Noise and Sparse Samples, epsilon = {}".format(control['epsilon']))
    plt.show()
    """

    # Create Gaussian process
    gpr = GaussianProcessRegressor()
    x_grid = x_grid.reshape(-1,1)
    x_learn = x_grid[x_sample]
    y_learn = noisy_truth[0,2,x_sample]
    gpr.fit(x_learn, y_learn)

    """
    for i in range(8):
        plt.plot(x_grid, gpr.sample_y(x_grid, random_state=i))
    #plt.legend()
    plt.xlabel("x-coordinate")
    plt.ylabel("Height Field")
    plt.title("Initial Conditions for Ensemble from Gaussian Process, epsilon = {}".format(control['epsilon']))
    plt.show()
    """

    # Produce initial conditions
    ICs = np.zeros((n_inits, 3, control['Nx']))
    for i in range(n_inits):
        ICs[i,2,:] = gpr.sample_y(x_grid, random_state=i)[:,0]

    # Pickle observations and ICs (securely?)
    assim_data = dict()
    assim_data['x_meas'] = x_sample
    assim_data['t'] = np.arange(0, (control['Nt']+1e-8)*control['timestep'], control['timestep'])
    assim_data['u'] = noisy_truth[:,:,x_sample]
    assim_data['ICs'] = ICs
    assim_data['size'] = n_inits

    with open('assim_data.dat', 'wb') as f:
        pickle.dump(assim_data, f)


if __name__ == "__main__":
    control_in = cyclops_control.setup_control(sys.argv[1:])
    main(control_in)
