import sys
import os
import pickle
import cyclops
import cyclops_control
import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

data_file = "test.dat"

control = dict()
Us = []

class U_data:
    def __init__(self, U_in):
        self.n_its = len(U_in)
        self.Nt = U_in[0].shape[0]
        self.Nx = U_in[0].shape[2]

        self.phys_data = np.zeros((self.n_its, self.Nt, 3, self.Nx))
        for i in range(self.n_its):
            self.phys_data[i,:,:,:] = U_in[i]

    def __str__(self):
        return "Physical Data"

    def get_height(self, it_num, timestep = None, spat_coord = None):
        if timestep is not None:
            return self.phys_data[it_num, timestep, 2, :]
        elif spat_coord is not None:
            return self.phys_data[it_num, :, 2, spat_coord]
        else:
            return self.phys_data[it_num, :, 2, :]

class MC_data:
    def __init__(self, control, it_ctr, U_in, rank):
        self.control = control
        self.n_eps = it_ctr.shape[0]
        self.n_T0 = it_ctr.shape[1]

        self.conv_data = it_ctr
        self.physical_data = []
        k = 0
        for i in range(self.n_eps):
            self.physical_data.append([])
            for j in range(self.n_T0):
                self.physical_data[i].append(U_data(U_in[k]))
                k += 1

    def get_run(self, eps, T0):
        eps_ind = self.control['epses'].index(eps)
        T0_ind = self.control['T0s'].index(T0)

        return self.physical_data[eps_ind][T0_ind]

    def __str__(self):
        return "Run on Rank {}".format(self.rank)

with open(data_file, "rb") as f:
    data_in = pickle.load(f)

control['epses'] = data_in['epses']
control['T0s'] = data_in['T0s']
control['timestep'] = data_in['timestep']
control['Nx'] = data_in['Nx']
control['size'] = data_in['size']

run_data = []
for i in range(control['size']):
    U = data_in['run_data'][i]['U']
    its = data_in['run_data'][i]['its']
    #gauss_width = data_in['run_data'][i]['sigma']

    run_data.append(MC_data(control, its, U, i))
    
mean_its = np.zeros((len(control['epses']), len(control['T0s'])))
i = 0
for eps in control['epses']:
    j = 0
    for T0 in control['T0s']:
        for mc_run in run_data:
            mean_its[i,j] += mc_run.get_run(eps, T0).phys_data.shape[0]
        j += 1
    i += 1
mean_its /= float(control['size'])

col_setup = ("c|" + len(control['epses'])*"c")
with open('conv_table.tex', 'w') as f:
    f.write("\\documentclass{standalone}\n")
    f.write("\\begin{document}\n")
    #f.write("\\begin{table}\n")
    f.write("\\centering\n")
    f.write("\\begin{tabular}{" + col_setup + "}\n")

    f.write("T0 &")
    for j in range(len(control['epses'])-1):
        f.write("$\\varepsilon$ = {} &".format(control['epses'][j]))
    f.write("$\\varepsilon$ = {} \\\\\n".format(control['epses'][-1]))
    f.write("\\hline\n")

    for i in range(len(control['T0s'])):
        f.write("{} &".format(control['T0s'][i]))
        for j in range(len(control['epses'])-1):
            f.write("{} &".format(mean_its[j,i]))
        f.write("{} \\\\\n".format(mean_its[len(control['epses']) - 1,i]))
        f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    #f.write("\\end{table}\n")
    f.write("\\end{document}\n")

"""
x_grid = np.linspace(0, 2*np.pi, control['Nx'])
for j in range(run_data[0].physical_data[0][0].Nt):
    for i in range(8):
        plt.plot(x_grid, run_data[i].get_run(control['epses'][0], control['T0s'][0]).get_height(-1, timestep = j))
        plt.ylim([-0.2,1.0])
    plt.show()
"""
