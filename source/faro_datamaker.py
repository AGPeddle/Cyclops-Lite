#!/usr/bin/env/python3
"""
An example script to set up an ensemble of initial conditions for the EnKF
"""

class Noiser:
    """
    A wrapper on a normal distribution to add white (in space and time) noise
    to a given vector.
    """

    def __init__(self, mean = 0, var = 0.05):
        self.mean = mean
        self.var = var

    def apply(self, data_in):
        """
        Adds white noise to a given vector
        """

        N = data_in.shape
        s = np.random.normal(self.mean, self.var, N)

        return data_in + s

def h_init(control):
    """
    This function sets up a two-humped Gaussian initial condition for the height field.
    """

    # Spatial domain
    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])

    # Height field
    h_space = np.exp(-2.0 * ((x_grid-control['Lx']/3.0)**2)) + \
              1.5*np.exp(-3.0 * ((x_grid-2.0*control['Lx']/3.0)**2))

    return h_space

def main(control):
    """
    Main program to create necessary data for a sequential data assimilation run
    with the faro script.
    """

    # Housekeeping
    if 'working_dir' in control: os.chdir(control['working_dir'])
    control['solver'] = "fine_propagator"
    station_spacing = control['meas_spacing']
    n_inits = control['ensemble_size']
    control['final_time'] = control['coarse_timestep']*control['assim_cycle_length']

    # Set up spatial domain
    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])
    x_sample = np.append(np.arange(0,x_grid.shape[0],station_spacing),-1)

    # Set up noiser object
    noiser = Noiser()

    # Create exponential integrator:
    expInt = ExponentialIntegrator(control)

    # Initialise spectral toolbox object
    st = SpectralToolbox(control['Nx'], control['Lx'])

    # Set up initial (truth) field
    truth = np.zeros((control['N_cycles'] + 1, 3, control['Nx']))
    truth[0, 2, :] = h_init(control)

    # Propagate it through N_cycles
    for i in range(control['N_cycles']):
        truth[i + 1, :, :] = RSWE_direct.solve(control, expInt, st, truth[i, :, :])

    # Noisify the data
    noisy_truth = noiser.apply(truth)

    # Create Gaussian process
    gpr = GaussianProcessRegressor()
    x_grid = x_grid.reshape(-1,1)
    x_learn = x_grid[x_sample]
    y_learn = noisy_truth[0,2,x_sample]
    x_learn = x_grid[np.array(([0,control['Nx']//3,2*control['Nx']//3,-1]))]
    y_learn = noisy_truth[0,2,np.array(([0,control['Nx']//3,2*control['Nx']//3,-1]))]
    gpr.fit(x_learn, y_learn)

    # Produce initial conditions
    ICs = np.zeros((n_inits, 3, control['Nx']))
    for i in range(n_inits):
        ICs[i,2,:] = gpr.sample_y(x_grid, random_state=i)[:,0]

    # Pickle observations and ICs
    assim_data = dict()
    assim_data['x_meas'] = x_sample
    assim_data['final_time'] = control['final_time']
    assim_data['t'] = np.arange(0, (control['N_cycles']+1e-8)*control['final_time'], control['final_time'])
    assim_data['u'] = noisy_truth[:,:,x_sample]
    assim_data['ICs'] = ICs
    assim_data['size'] = n_inits

    with open('{}_assim_data.dat'.format(control['outFileStem']), 'wb') as f:
        pickle.dump(assim_data, f)

if __name__ == "__main__":
    control_in = cyclops_control.setup_control(sys.argv[1:])
    main(control_in)
