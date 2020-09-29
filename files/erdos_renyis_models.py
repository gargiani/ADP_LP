import numpy as np
import control
import os
import json

'''
This script creates and saves in json files LTI systems using the Erdos-Renyi model.
'''

#set random seed
np.random.seed(0)

N_x = [i for i in range(2, 52, 2)]

#parameters for realization of Erdos-Renyi's graph
p = 0.7
range_u = 0.1#0.01
range_b = 0.1
rank_CM = 0
rank_OM = 0

directory = '../data'

if not os.path.exists(directory):
    os.makedirs(directory)

for n_x in N_x:
    #set input to 10% of states number

    if n_x <= 10:
        N_u = 2
    else:
        N_u = 3

    #numbers of measurements is the same as the number of states
    N_y = n_x

    while rank_OM<n_x or rank_CM<n_x:#we require observability and controllability

        A = 0.5*np.eye(n_x)

        for ii in range(n_x):
            for jj in range(n_x):
                if ii!=jj:
                    A[ii, jj] = np.random.choice([0,1], p=[1-p, p])*np.random.uniform(-range_u, range_u)

        B = np.zeros((n_x, N_u))

        for ii in range(n_x):
            for jj in range(N_u):
                B[ii,jj] = np.random.choice([0,1], p=[1-p, p])*np.random.uniform(-range_b, range_b)
                controllability_matrix = control.ctrb(A, B)

        C = np.eye(max(n_x, N_y))
        C = C[0:N_y, 0:n_x]

        observability_matrix = control.obsv(A, C)

        rank_CM = np.linalg.matrix_rank(controllability_matrix)#, tol=1e-16)
        rank_OM = np.linalg.matrix_rank(observability_matrix)#, tol=1e-16)

        print('number of states {}, controllability matrix rank {}, observability matrix rank {}'.format(n_x, rank_CM, rank_OM))

    data = {'A': A.tolist(), 'B': B.tolist(), 'C': C.tolist()}

    f_name = '{}_states.json'.format(n_x)

    with open(os.path.join(directory, f_name), 'w') as f:
        json.dump(data, f)

    f.close()
