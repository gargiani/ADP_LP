from dynamical_systems import pendulum
#from ADP_LP.methods import Qstar_LP, Qhat_LP
from methods import Qstar_LP, Qhat_LP
from ADP_LP.policies import linear_policy
import numpy as np
import json
import os
import torch
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

type = torch.float64

parser = argparse.ArgumentParser(description='Solving CartPole with LP approach.')

parser.add_argument('-LP_approach', type=int, help='1 for Qstar LP, >=2 \
for Qhat-LP', default=1)
parser.add_argument('-LP_solver', type=str, help='LP solver', default='GUROBI')
parser.add_argument('-samples', type=int, help='number of samples for \
empirical mean', default=1)
parser.add_argument('-u_range', type=float, help='range for u sampling', default=10.0)
parser.add_argument('-x_range', type=float, nargs='+', help='range for x sampling', default=0.01)
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
parser.add_argument('-m', type=float, help='point mass of the \
rod, [kg]', default=2)
parser.add_argument('-l', type=float, help='lenght \
of the rod, [m]', default=1)
parser.add_argument('-k', type=float, help='friction coeffiecient, [kg/(m*s)]', default=0)
parser.add_argument('-delta_t', type=float, help='sampling time', default=1)
parser.add_argument('-verbose', type=int, help='solver verbose', default=1)

args = parser.parse_args()

res_dir = '../results_pendulum'

torch.manual_seed(args.random_seed)

m = args.m
l = args.l
k = args.k

n_x = 2
delta_t = args.delta_t
C = torch.eye(n_x, dtype=type)
rho = 0.95
gamma = 0.95
std_dev = args.std_dev

x_ranges = [[-ii, ii] for ii in args.x_range]

sys = pendulum(m, l, k, delta_t, C, rho, gamma, std_dev)


#state-action relevance weights, see http://www.mit.edu/~pucci/discountedLP.pdf
XU_weights = torch.cat((torch.ones(sys.N_x, dtype=type), 0.8*torch.ones(sys.N_u, dtype=type)), 0)

#optimization method
if args.LP_approach == 1:
    method = Qstar_LP(sys, XU_weights, linear_policy, args.LP_solver, verbose=args.verbose)
    eps = [args.eps]
else:
    method = Qhat_LP(sys, XU_weights, linear_policy, args.LP_solver, verbose=args.verbose)
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
    M_0 = 0.001*torch.ones((sys.N_u, sys.N_x), dtype=type)

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
    if args.save:
        res = {'M': M, 'obj_fun': E_Q,
               'time': time,'Q_q': Q_q.tolist(),'e_q': e_q.tolist(),'rho':rho,\
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
        res = {'state':'unbounded',\
               'time': time,'rho':rho,'gamma': gamma, 'sigma': std_dev,\
               'c': XU_weights.tolist(), 'x_range':args.x_range}
        if args.exp_sampling:
            res['u_range'] = args.u_range
        else:
            res['M_0'] = M_0.tolist()
            res['eps'] = eps

if args.LP_approach == 1:
    try:
        print('E_Q {}, solver time [s] {}'.format(E_Q, time))
    except:
        print('ops! looks like your problem is unbounded! solver time [s] {}'.format(time))
elif args.LP_approach == 2:
    try:
        print('E_Q {}, solver time [s] {}'.format(E_Q, time))
    except:
        print('ops! looks like your problem is unbounded! solver time [s] {}'.format(time))

if args.save:
    #saving results
    if args.LP_approach == 1:
        root_directory = 'Q_star_LP'

    elif args.LP_approach == 2:
        root_directory = 'Q_hat_LP'

    if not os.path.exists(os.path.join(res_dir, root_directory)):
        os.makedirs(os.path.join(res_dir, root_directory))

    if args.exp_sampling:

        f_name = '{}_states_{}_constraint_{}_sigma_{}_samples_{}_xrange_{}_urange_{}_symmetric_{}_run.json'.format(sys.N_x, args.n_constraints, std_dev, args.samples, args.x_range, args.u_range, args.symmetric, args.random_seed)

    else:

        f_name = '{}_states_{}_constraint_{}_sigma_{}_samples_{}_xrange_{}_eps_{}_symmetric_{}_run.json'.format(sys.N_x, args.n_constraints, std_dev, args.samples, args.x_range, args.eps, args.symmetric, args.random_seed)

    with open(os.path.join(res_dir, root_directory, f_name), 'w') as f:

        json.dump(res, f)

    f.close()
