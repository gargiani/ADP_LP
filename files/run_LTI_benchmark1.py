from ADP_LP.methods import Qstar_LP, Qhat_LP
from ADP_LP.policies import linear_policy
from ADP_LP.dynamical_systems import dlqr
import numpy as np
import json
import os
import torch
import argparse
from auxiliaries import str2bool

type = torch.float64

parser = argparse.ArgumentParser(description='Solving LQR with LP approach.')

parser.add_argument('-LP_approach', type=int, help='1 for Qstar LP, >=2 \
for Qhat-LP', default=1)
parser.add_argument('-LP_solver', type=str, help='LP solver', default='GUROBI')
parser.add_argument('-samples', type=int, help='number of samples for \
empirical mean', default=1)
parser.add_argument('-n_x', type=int, help='number of states', default=10)
parser.add_argument('-u_range', type=float, help='range for u sampling', default=10.0)
parser.add_argument('-x_range', type=float, nargs='+', help='range for x sampling', default=[0.01])
parser.add_argument('-std_dev', type=float, help='standard deviation for noise \
affecting the dynamical system', default=0.0)
parser.add_argument('-symmetric', type=str2bool, help='space of symmetric or\
non symmetric matrices', default=True)
parser.add_argument('-exp_sampling', type=str2bool, help='explorative or non\
 explorative sampling.', default=True)
parser.add_argument('-random_seed', type=int, help='random_seed', default=0)
parser.add_argument('-n_constraints', type=int, help='number of sampled \
constraints', default=1000)
parser.add_argument('-save', type=str2bool, help='save results', default=True)
parser.add_argument('-eps', type=float, help='noise corruption \
affecting action', default=0.0)
parser.add_argument('-verbose', type=int, help='solver verbose', default=1)

args = parser.parse_args()

res_dir = '../results_benchmark1'

#import the dynamical system: x_{k+1} = A*x_{k} + B*u_{k} + w_{k}, l(x, u) = x^T*C^T*C*x + rho*u^T*u
directory = '../data'#'Erdos_Renyi_LTI_systems'

torch.manual_seed(args.random_seed)

f_name = '{}_states.json'.format(args.n_x)

with open(os.path.join(directory, f_name), 'r') as f:
    res = json.load(f)

f.close()

A = torch.tensor(res['A'], dtype=type)
B = torch.tensor(res['B'], dtype=type)
C = torch.tensor(res['C'], dtype=type)

N_u = B.shape[1]

print('number of states {}, number of inputs {}'.format(args.n_x, N_u))

rho = 0.01
gamma = 0.95 #discount factor
std_dev = args.std_dev

LQR = dlqr(A, B, C, rho, gamma, sigma=std_dev)

#state-action relevance weights, see http://www.mit.edu/~pucci/discountedLP.pdf
XU_weights = torch.cat((torch.ones(args.n_x, dtype=type), 0.8*torch.ones(N_u, dtype=type)), 0)

#compute LQR solutions via DARE
P_opt, M_opt, e_star = LQR.optimal_solution()
Q_star, E_Qstar, gap = LQR.optimal_q(P_opt, e_star, XU_weights)

if len(args.x_range)==args.n_x:
    centers = args.n_x*[0]
elif len(args.x_range)==1:
    centers = [0]
else:
    raise Exception('choose an appropriate number of ranges for the state intialization.')

x_ranges = [[-args.x_range[min(len(args.x_range)-1, jj)]+center_ii, args.x_range[min(len(args.x_range)-1, jj)]+center_ii] for jj,center_ii in enumerate(centers)]

#optimization method
if args.LP_approach == 1:
    method = Qstar_LP(LQR, XU_weights, linear_policy, args.LP_solver, verbose=args.verbose)
    eps = [args.eps]
else:
    method = Qhat_LP(LQR, XU_weights, linear_policy, args.LP_solver, verbose=args.verbose)
    eps = [args.eps, args.eps]

#refresh memory of method
method.A_memory = None
method.b_memory = None

if args.LP_approach == 1:
    n_constraints = int(args.n_constraints/2)
else:
    n_constraints = int(args.n_constraints)

print('policy buffer')

if args.exp_sampling:
    X, U, Xplus, L_buffer, W, Uprime = method.buffer(n_constraints, args.samples,
    X_space=x_ranges, U_space=[-args.u_range, args.u_range])
else:
    #exploration policy
    M_0 = M_opt + 0.2*torch.ones((N_u, args.n_x), dtype=type)

    X, U, Xplus, L_buffer, W, Uprime = method.buffer(n_constraints, args.samples,
    M=M_0, X_space=x_ranges, epsilon=eps)

print('policy evaluation')

if args.symmetric:
    A, b, c, Q_q, e_q, Q_v, e_v, E_Q, x_star, time = \
    method.policy_evaluation_symm(X, U, L_buffer, Xplus, Uprime)
else:
    A, b, c, Q_q, e_q, Q_v, e_v, E_Q, x_star, time  = \
    method.policy_evaluation(X, U, L_buffer, Xplus, Uprime)

print('extract the greedy policy')

if x_star is not None:
    M = method.greedy_policy(Q_q)
    M = M.tolist()
    print("lqr policy {}, opt. policy {}".format(M_opt.tolist(), M))
    if args.save:
        res = {'M': M, 'obj_fun': E_Q, 'obj_fun_star': E_Qstar,\
               'time': time,'M_star': M_opt.tolist(), 'gap': gap.tolist(),\
               'Q_q': Q_q.tolist(),'e_q': e_q.tolist(),'rho':rho,\
               'gamma': gamma, 'sigma': std_dev,\
               'c': XU_weights.tolist(), 'x_range':args.x_range, 'state':'solved'}
        if args.exp_sampling:
            res['u_range'] = args.u_range

        else:
            res['M_0'] = M_0.tolist()
            res['eps'] = eps

else:
    M = None
    if args.save:
        res = {'state':'unbounded', 'obj_fun_star': E_Qstar,\
               'time': time,'M_star': M_opt.tolist(), 'gap': gap.tolist(),\
               'rho':rho,'gamma': gamma, 'sigma': std_dev,\
               'c': XU_weights.tolist(), 'x_range':args.x_range}
        if args.exp_sampling:
            res['u_range'] = args.u_range
        else:
            res['M_0'] = M_0.tolist()
            res['eps'] = eps

if args.LP_approach == 1:
    try:
        print('optimality gap {}, solver time [s] {}'.format(np.abs(E_Q-E_Qstar), time))
    except:
        print('ops! looks like your problem is unbounded! solver time [s] {}'.format(time))
elif args.LP_approach == 2:
    try:
        print('optimality gap {}, gap {}, solver time [s] {}'.format(np.abs(E_Q-gap-E_Qstar), gap, time))
    except:
        print('ops! looks like your problem is unbounded! solver time [s] {}'.format(time))

if args.save:
    #saving results
    import pdb; pdb.set_trace()
    if args.LP_approach == 1:
        root_directory = 'Q_star_LP'

    elif args.LP_approach == 2:
        root_directory = 'Q_hat_LP'

    if not os.path.exists(os.path.join(res_dir, root_directory)):
        os.makedirs(os.path.join(res_dir, root_directory))

    if args.exp_sampling:

        f_name = '{}_states_{}_constraint_{}_sigma_{}_samples_{}_xrange_{}_urange_{}_symmetric_{}_run.json'.format(args.n_x, args.n_constraints, std_dev, args.samples, args.x_range, args.u_range, args.symmetric, args.random_seed)

    else:

        f_name = '{}_states_{}_constraint_{}_sigma_{}_samples_{}_xrange_{}_eps_{}_symmetric_{}_run.json'.format(args.n_x, args.n_constraints, std_dev, args.samples, args.x_range, args.eps, args.symmetric, args.random_seed)

    with open(os.path.join(res_dir, root_directory, f_name), 'w') as f:

        json.dump(res, f)

    f.close()
