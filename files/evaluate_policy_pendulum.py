from ADP_LP.dynamical_systems import pendulum
from ADP_LP.policies import linear_policy

import numpy as np
import json
import os
import torch

import matplotlib.pyplot as plt

type = torch.float64

samples = 100
symm = True
n_constraints = 100000
u_range = 10.0#1000.0
x_range = 1.0
n_x = 2
run = 0
std_dev = 0.0001

m = 2
k = 0
l = 1

delta_t = 0.01
C = torch.eye(n_x, dtype=type)
rho = 0.95
gamma = 0.95

sys = pendulum(m, l, k, delta_t, C, rho, gamma, std_dev)

methods = ['Q_hat_LP', 'Q_star_LP']

root_directory = '../results_pendulum'#'/home/matilde/ADP_LP/results_cart_pole/'

f_name = '{}_states_{}_constraint_{}_sigma_{}_samples_{}_xrange_{}_urange_{}_symmetric_{}_run.json'.format(n_x, n_constraints, std_dev, samples, x_range, u_range, symm, run)

#initialization
N = 10
x00_range = 0.1
x01_range = 0.1
# initialization
X00 =  (-x00_range - x00_range)*torch.rand((N, 1, 1), dtype=type) + x00_range
X01 =  (-x01_range - x01_range)*torch.rand((N, 1, 1), dtype=type) + x01_range
X_init = torch.cat((X00, X01), 1)

time_horizon = 2500

for jj, method in enumerate(methods):

    print(method)
    tmp1 = torch.tensor([], dtype=torch.float64)
    tmp2 = torch.tensor([], dtype=torch.float64)

    with open(os.path.join(root_directory, method, f_name), 'r') as f:
        res = json.load(f)
    f.close()

    M = torch.tensor(res['M'], dtype=type)

    X = X_init

    J = torch.zeros((N, 1, 1), dtype=type)

    for ii in range(time_horizon):

        U = linear_policy(M, X, epsilon=None)

        X_plus, cost, W = sys.simulate(1, X, U)
        X = X_plus.squeeze(1)

        tmp1 = torch.cat((tmp1, X[:, 0, :]), 1)
        tmp2 = torch.cat((tmp2, X[:, 1, :]), 1)

        J += (gamma**ii)*cost

    J_mean = J.mean(axis = 0)

    if method == 'Q_hat_LP':
        plt.figure(1)
        for n in range(N):
            if n==0:
                plt.plot(np.arange(time_horizon), tmp1[n, :], c='r', label=r'$x_1$')
                plt.plot(np.arange(time_horizon), tmp2[n, :], c='b', label=r'$x_2$')
            else:
                plt.plot(np.arange(time_horizon), tmp1[n, :], c='r')
                plt.plot(np.arange(time_horizon), tmp2[n, :], c='b')

        plt.title(method+" [{:.3f} s]".format(res['time'])+",  "+r"$E_{x_1, x_2\sim P_0}\left[\,\hat{J}(x_1, x_2)\,\right]=$"+"{:.3f}".format(J_mean.item()))
        plt.xlabel(r'N')
        plt.ylabel(r'$x_1, \,\,\,x_2$')
        plt.grid(True)
        plt.legend()
    else:
        plt.figure(2)
        for n in range(N):
            if n==0:
                plt.plot(np.arange(time_horizon), tmp1[n, :], c='r', label=r'$x_1$')
                plt.plot(np.arange(time_horizon), tmp2[n, :], c='b', label=r'$x_2$')
            else:
                plt.plot(np.arange(time_horizon), tmp1[n, :], c='r')
                plt.plot(np.arange(time_horizon), tmp2[n, :], c='b')
        plt.title(method+" [{:.3f} s]".format(res['time'])+",  "+r"$E_{x_1, x_2\sim P_0}\left[\,\tilde{J}(x_1, x_2)\,\right]=$"+"{:.3f}".format(J_mean.item()))
        plt.xlabel(r'N')
        plt.ylabel(r'$x_1, \,\,\,x_2$')
        plt.grid(True)
        plt.legend()
plt.show()
