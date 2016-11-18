#!/bin/python3

import sys
import os
import pickle
import cyclops
import cyclops_control
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

control_in = cyclops_control.setup_control(sys.argv[1:])

mu, sigma = 2.0, 0.1
gauss_width = np.random.normal(mu, sigma)

cyclops.main(control_in, rank, gauss_width)

comm.Barrier()
if rank == 0:
    out_data = dict()
    with open("{}_0.dat".format(control_in['outFileStem']), 'rb') as f:
        data_in = pickle.load(f)

        out_data['epses'] = data_in['epses']
        out_data['T0s'] = data_in['T0s']
        out_data['timestep'] = data_in['timestep']
        out_data['Nx'] = data_in['Nx']
        out_data['size'] = size

    run_data = []
    for i_rank in range(size):
        filename = "{}_{}.dat".format(control_in['outFileStem'], i_rank)
        with open(filename, 'rb') as f:
            data_in = pickle.load(f)

            data_buff = dict()
            data_buff['U'] = data_in['U_space']
            data_buff['its'] = data_in['it_ctr']
            data_buff['sigma'] = data_in['gauss_width']
            run_data.append(data_buff)

        os.remove(filename)
    out_data['run_data'] = run_data

    with open("{}.dat".format(control_in['outFileStem']), 'wb') as f:
        pickle.dump(out_data, f)

    print("Alles Gute")
