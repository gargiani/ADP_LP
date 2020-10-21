from ADP_LP.dynamical_systems import cart_pole
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
u_range = 100.0#1000.0
x_range = [5.0, 5.0, 1.0, 2.0]
n_x = 4
run = 0
std_dev = 0.0001

m_p = 2
m_c = 4
l = 1

delta_t = 0.1
C = torch.eye(n_x, dtype=type)
rho = 0.95
gamma = 0.95

sys = cart_pole(m_c, m_p, l, delta_t, C, rho, gamma, std_dev)

methods = ['Q_hat_LP', 'Q_star_LP']

root_directory = '../results_cart_pole'#'/home/matilde/ADP_LP/results_cart_pole/'

f_name = '{}_states_{}_constraint_{}_sigma_{}_samples_{}_xrange_{}_urange_{}_symmetric_{}_run.json'.format(n_x, n_constraints, std_dev, samples, x_range, u_range, symm, run)

#initialization
N = 10
centers = [0, 0, 0, 0]
delta_x = [0.01, 0.01, 0.01, 0.01]
x0_ranges = [[-ii+centers[jj], ii+centers[jj]] for jj,ii in enumerate(delta_x)]

X_init = torch.tensor([], dtype=type)

# initialization
for n in range(n_x):
    X_ii = (x0_ranges[n][0] - x0_ranges[n][1])*torch.rand((N, 1, 1), dtype=type) + x0_ranges[n][1]
    X_init = torch.cat((X_init, X_ii), 1)

time_horizon = 1000

for jj, method in enumerate(methods):

    print(method)
    import pdb; pdb.set_trace()
    tmp1 = torch.tensor([], dtype=torch.float64)
    tmp2 = torch.tensor([], dtype=torch.float64)
    tmp3 = torch.tensor([], dtype=torch.float64)
    tmp4 = torch.tensor([], dtype=torch.float64)

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
        tmp3 = torch.cat((tmp3, X[:, 2, :]), 1)
        tmp4 = torch.cat((tmp4, X[:, 3, :]), 1)


        J += (gamma**ii)*cost

    J_mean = J.mean(axis = 0)

    if method == 'Q_hat_LP':
        plt.figure(1)
        for n in range(N):
            if n==0:
                plt.plot(np.arange(time_horizon), tmp1[n, :], c='r', label=r'$x_1$')
                plt.plot(np.arange(time_horizon), tmp2[n, :], c='b', label=r'$x_2$')
                plt.plot(np.arange(time_horizon), tmp3[n, :], c='m', label=r'$x_3$')
                plt.plot(np.arange(time_horizon), tmp4[n, :], c='g', label=r'$x_4$')
            else:
                plt.plot(np.arange(time_horizon), tmp1[n, :], c='r')
                plt.plot(np.arange(time_horizon), tmp2[n, :], c='b')
                plt.plot(np.arange(time_horizon), tmp3[n, :], c='m')
                plt.plot(np.arange(time_horizon), tmp4[n, :], c='g')

        plt.title(method+" [{:.3f} s]".format(res['time'])+",  "+r"$E_{x_1, x_2, x_3, x_4\sim P_0}\left[\,\hat{J}(x_1, x_2, x_3, x_4)\,\right]=$"+"{:.3f}".format(J_mean.item()))
        plt.xlabel(r'N')
        plt.ylabel(r'$x_1, \,\,\,x_2, \,\,\,x_3, \,\,\,x_4$')
        plt.grid(True)
        plt.legend()
    else:
        plt.figure(2)
        for n in range(N):
            if n==0:
                plt.plot(np.arange(time_horizon), tmp1[n, :], c='r', label=r'$x_1$')
                plt.plot(np.arange(time_horizon), tmp2[n, :], c='b', label=r'$x_2$')
                plt.plot(np.arange(time_horizon), tmp3[n, :], c='m', label=r'$x_3$')
                plt.plot(np.arange(time_horizon), tmp4[n, :], c='g', label=r'$x_4$')
            else:
                plt.plot(np.arange(time_horizon), tmp1[n, :], c='r')
                plt.plot(np.arange(time_horizon), tmp2[n, :], c='b')
                plt.plot(np.arange(time_horizon), tmp3[n, :], c='m')
                plt.plot(np.arange(time_horizon), tmp4[n, :], c='g')
        plt.title(method+" [{:.3f} s]".format(res['time'])+",  "+r"$E_{x_1, x_2, x_3, x_4\sim P_0}\left[\,\tilde{J}(x_1, x_2, x_3, x_4)\,\right]=$"+"{:.3f}".format(J_mean.item()))
        plt.xlabel(r'N')
        plt.ylabel(r'$x_1, \,\,\,x_2, \,\,\,x_3, \,\,\,x_4$')
        plt.grid(True)
        plt.legend()
plt.show()
