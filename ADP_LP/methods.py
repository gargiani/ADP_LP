import scipy.optimize
import torch
import time
import gurobipy as gp
from gurobipy import GRB
from ADP_LP.policies import linear_policy

type = torch.float64

class LP_approach:

    def __init__(self, lqr, Sigma, pi, LP_solver='scipy', verbose=1):
        if LP_solver not in ['GUROBI', 'scipy']:
            LP_solver = 'scipy'
            raise Warning('The selected LP solver is not available.\
                           scipy LP solver will be used instead.')
        self.lqr = lqr
        self.Sigma = Sigma
        self.policy = pi
        self.A_memory = None
        self.b_memory = None
        self.LP_solver = LP_solver
        self.verbose = verbose

    def __call_solver__(self, A, b, c):

        if self.A_memory is not None:

            A = torch.cat([self.A_memory, A], 0)
            b = torch.cat([self.b_memory, b], 0)

        self.A_memory = A
        self.b_memory = b

        print('number of variables: {}, number of constraints: {}'.format(A.shape[1], A.shape[0]))

        if self.LP_solver == 'scipy':
            print('calling the LP solver...')

            start_time = time.time()

            res = scipy.optimize.linprog(c.tolist(), A_ub=A.tolist(), b_ub=b.tolist(),\
                                         bounds=(None, None), method='interior-point',\
                                         options={'cholesky':False, 'sym_pos':False,\
                                         'lstsq':False, 'maxiter': 5000, 'tol': 1e-6})
            end_time = time.time()

            if not res.success:
                if res.status==1:
                    raise Exception('iterations limit reached.')

                if res.status==2:
                    raise Exception('the problem appears to be infeasible.')

                if res.status==3:
                    raise Exception('the problem appears to be unbounded.')

                if res.status==4:
                    raise Exception('numerical difficulties encountered.')

            else:
                print('solved!')

            return res.x, -res.fun, end_time-start_time

        else:
            try:

                m = gp.Model()

                #set model parameters
                if self.verbose == 0:
                    m.setParam('OutputFlag', 0)

                m.setParam('DualReductions', 0)
                m.setParam('FeasibilityTol', 1e-9)
                m.setParam('OptimalityTol', 1e-9)

                x = m.addMVar(shape=c.numpy().shape[0], lb=-GRB.INFINITY,\
                              ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='x')
                m.setObjective(c.numpy() @ x, GRB.MINIMIZE)
                m.addConstr(A.numpy() @ x <= b.numpy(), name='constraints')

                print('calling the LP solver...')

                start_time = time.time()

                m.optimize()

                end_time = time.time()

                if m.Status == 2: #the problem has been successfully solved
                    print('problem solved :)')
                    return x.X, -m.objVal, end_time-start_time
                elif m.Status == 5: #the problem is unbounded
                    print('the problem is unbounded.')
                    return None, None, end_time-start_time
                else:
                    print('the problem is infeasible.')
                    return None, None, end_time-start_time
            except:
                raise Exception('An error occurred while GUROBI was trying to solve the LP.')

    def __sampling__(self, P, K, M, X_space, U_space, epsilon):

        X_buffer = (X_space[0] - X_space[1])*torch.rand((P, self.lqr.N_x, 1), dtype=type) + X_space[1]

        if M==None:
            print('using explorative roll-outs')
            U_buffer = (U_space[0] - U_space[1])*torch.rand((P, self.lqr.N_u, 1), dtype=type) + U_space[1]
        else:
            print('unsing linear policy for roll-outs')
            U_buffer = self.policy(M, X_buffer, epsilon)

        Xplus_buffer, L_buffer, W = self.lqr.simulate(K, X_buffer, U_buffer)

        return X_buffer, U_buffer, Xplus_buffer, L_buffer, W

    def greedy_policy(self, S):

        C_x = -(S[self.lqr.N_x:, :self.lqr.N_x] + \
              torch.transpose(S[:self.lqr.N_x, self.lqr.N_x:], 0, 1))
        J_u = S[self.lqr.N_x:, self.lqr.N_x:] + \
              torch.transpose(S[self.lqr.N_x:, self.lqr.N_x:], 1, 0)

        try:
            M = torch.matmul(torch.inverse(J_u), C_x)
        except:
            M = torch.matmul(torch.pinverse(J_u), C_x)
        return M

class Qstar_LP(LP_approach):

    """Data-Driven Q-learning with LP approach."""

    def __init__(self, lqr, Sigma, pi, LP_solver='scipy', verbose=1):

        super().__init__(lqr, Sigma, pi, LP_solver, verbose)
        #self.N_var = (self.lqr.N_x+self.lqr.N_u)**2 + self.lqr.N_x**2 + 2

    def buffer(self, P, K, M=None, X_space=[-10, 10], U_space=[-10, 10], epsilon=[None]):

        X_buffer, U_buffer, Xplus_buffer, L_buffer, W = \
        self.__sampling__(P, K, M, X_space, U_space, epsilon[0])

        return X_buffer, U_buffer, Xplus_buffer, L_buffer, W, None

    def policy_evaluation(self, X, U, L, Xplus, Uprime):

        XU = torch.cat((X, U), 1)

        A_Q = torch.matmul(XU, XU.transpose(1,2))

        A_Q = torch.cat((torch.reshape(A_Q, (A_Q.shape[0], (self.lqr.N_x+self.lqr.N_u)**2, )),\
                         torch.ones((A_Q.shape[0], 1), dtype=type)), -1)

        Aplus_V = self.lqr.gamma*torch.mean(torch.matmul(Xplus, Xplus.transpose(2, 3)), 1)

        Aplus_V = torch.cat((torch.reshape(Aplus_V, (Aplus_V.shape[0], self.lqr.N_x**2, )),\
                             self.lqr.gamma*torch.ones((Aplus_V.shape[0], 1), dtype=type)), -1)

        A_V = torch.matmul(X, X.transpose(1,2))

        A_V = torch.cat((torch.reshape(A_V, (A_V.shape[0], self.lqr.N_x**2, )),\
                         torch.ones((A_V.shape[0], 1), dtype=type)), -1)

        #A = [[A_Q, -Aplus_V], [-A_Q, A_V]]
        A = torch.cat((torch.cat((A_Q, -Aplus_V), 1), torch.cat((-A_Q, A_V), 1)), 0)

        b = torch.cat((L.flatten(), torch.zeros(A_Q.shape[0], dtype=type)))

        c = -torch.cat((torch.reshape(torch.diag_embed(self.Sigma),\
                      ((self.lqr.N_x+self.lqr.N_u)**2, 1)),\
                       torch.tensor([[1.]], dtype=type)), 0)

        c = torch.cat((c.flatten(), torch.zeros(self.lqr.N_x**2+1, dtype=type)))

        x_star, obj_fun, time = self.__call_solver__(A, b, c)

        if x_star is not None:

            S = torch.tensor(x_star, dtype=type).unsqueeze(1)

            Q_q = torch.reshape(S[:(self.lqr.N_u + self.lqr.N_x)**2], \
                               (self.lqr.N_u + self.lqr.N_x, self.lqr.N_u + self.lqr.N_x))

            e_q = S[(self.lqr.N_u + self.lqr.N_x)**2:(self.lqr.N_u + self.lqr.N_x)**2+1]

            Q_v = torch.reshape(S[(self.lqr.N_u + self.lqr.N_x)**2+1:(self.lqr.N_u \
                                  + self.lqr.N_x)**2 + 1 + self.lqr.N_x**2], \
                                  (self.lqr.N_x, self.lqr.N_x))

            e_v = S[(self.lqr.N_u + self.lqr.N_x)**2 + 1 + self.lqr.N_x**2:\
                    (self.lqr.N_u + self.lqr.N_x)**2 + 1 + self.lqr.N_x**2+1]

            return A, b, c, Q_q, e_q, Q_v, e_v, obj_fun, x_star, time

        else:

            return A, b, c, None, None, None, None, obj_fun, x_star, time

    def policy_evaluation_symm(self, X, U, L, Xplus, Uprime):

        XU = torch.cat((X, U), 1)

        A_Q = torch.matmul(XU, XU.transpose(1,2))

        mask = torch.cat(A_Q.shape[0]*[torch.eye(A_Q.shape[1], A_Q.shape[2], dtype=torch.bool).unsqueeze(0)], 0)
        A_Q = A_Q + A_Q.clone().masked_fill_(mask, 0)

        #extract here the upper triangular part
        A_Q = A_Q[:,torch.triu(torch.ones(A_Q.shape[1], A_Q.shape[2]))==1]

        A_Q = torch.cat((A_Q, torch.ones((A_Q.shape[0], 1), dtype=type)), -1)

        Aplus_V = self.lqr.gamma*torch.mean(torch.matmul(Xplus, Xplus.transpose(2, 3)), 1)

        mask = torch.cat(Aplus_V.shape[0]*[torch.eye(Aplus_V.shape[1], Aplus_V.shape[2], dtype=torch.bool).unsqueeze(0)], 0)
        Aplus_V = Aplus_V + Aplus_V.clone().masked_fill_(mask, 0)

        #extract here the upper triangular part
        Aplus_V = Aplus_V[:,torch.triu(torch.ones(Aplus_V.shape[1], Aplus_V.shape[2]))==1]

        Aplus_V = torch.cat((Aplus_V, self.lqr.gamma*torch.ones((Aplus_V.shape[0], 1), dtype=type)), -1)

        A_V = torch.matmul(X, X.transpose(1,2))

        mask = torch.cat(A_V.shape[0]*[torch.eye(A_V.shape[1], A_V.shape[2], dtype=torch.bool).unsqueeze(0)], 0)
        A_V = A_V + A_V.clone().masked_fill_(mask, 0)

        #extract here the upper triangular part
        A_V = A_V[:,torch.triu(torch.ones(A_V.shape[1], A_V.shape[2]))==1]

        A_V = torch.cat((A_V, torch.ones((A_V.shape[0], 1), dtype=type)), -1)

        A = torch.cat((torch.cat((A_Q, -Aplus_V), -1), torch.cat((-A_Q, A_V), -1)), 0)

        b = torch.cat((L.flatten(), torch.zeros(A_Q.shape[0], dtype=type)))

        Sigma_diag = torch.diag_embed(self.Sigma)

        c = -torch.cat((Sigma_diag[torch.triu(torch.ones(Sigma_diag.shape[0], Sigma_diag.shape[1]))==1],\
                       torch.tensor([1.], dtype=type)), 0)

        c = torch.cat((c.flatten(), torch.zeros(int(self.lqr.N_x*(self.lqr.N_x+1)/2)+1, dtype=type)))

        x_star, obj_fun, time = self.__call_solver__(A, b, c)

        if x_star is not None:

            S_Q = torch.tensor(x_star, dtype=type)[:int((self.lqr.N_u + self.lqr.N_x)*(self.lqr.N_u + self.lqr.N_x+1)/2)+1]
            S_V = torch.tensor(x_star, dtype=type)[int((self.lqr.N_u + self.lqr.N_x)*(self.lqr.N_u + self.lqr.N_x+1)/2)+1:]

            Q_q = torch.zeros(self.lqr.N_u + self.lqr.N_x, self.lqr.N_u + self.lqr.N_x, dtype=type)

            Q_q[torch.triu(torch.ones(self.lqr.N_u + self.lqr.N_x, self.lqr.N_u + self.lqr.N_x, dtype=type))==1] = S_Q[:-1]
            mask = torch.eye(self.lqr.N_u + self.lqr.N_x, self.lqr.N_u + self.lqr.N_x, dtype=torch.bool)
            Q_q = Q_q + Q_q.clone().transpose(0,1).masked_fill_(mask, 0)

            Q_v = torch.zeros(self.lqr.N_x, self.lqr.N_x, dtype=type)

            Q_v[torch.triu(torch.ones(self.lqr.N_x, self.lqr.N_x, dtype=type))==1] = S_V[:-1]
            mask = torch.eye(self.lqr.N_x, self.lqr.N_x, dtype=torch.bool)
            Q_v = Q_v + Q_v.clone().transpose(0,1).masked_fill_(mask, 0)

            return A, b, c, Q_q, S_Q[-1], Q_v, S_V[-1], obj_fun, x_star, time

        else:

            return A, b, c, None, None, None, None, obj_fun, x_star, time

class Qhat_LP(LP_approach):

    """Data-Driven Q-learning with LP approach on the simplified operator."""

    def __init__(self, lqr, Sigma, pi, LP_solver='scipy', verbose=1):

        super().__init__(lqr, Sigma, pi, LP_solver, verbose)
        #self.N_var = (self.lqr.N_x+self.lqr.N_u)**2 + 1

    def buffer(self, P, K, M=None, X_space=[-10, 10], U_space=[-10, 10], epsilon=[None, None]):

        X_buffer, U_buffer, Xplus_buffer, L_buffer, W = \
        self.__sampling__(P, K, M, X_space, U_space, epsilon[0])

        if M==None:
            Uprime_buffer = (U_space[0] - U_space[1])*torch.rand((P, self.lqr.N_u, 1), dtype=type) + U_space[1]
            Uprime_buffer = torch.cat((K*[Uprime_buffer.unsqueeze(1)]), 1)
        else:
            Uprime_buffer = self.policy(M, Xplus_buffer, epsilon[1])

        return X_buffer, U_buffer, Xplus_buffer, L_buffer, W, Uprime_buffer

    def policy_evaluation(self, X, U, L, Xplus, Uprime):

        XU = torch.cat((X, U), 1)

        XU_prime = torch.cat((Xplus, Uprime), 2)

        A = torch.matmul(XU, XU.transpose(1,2)) -\
            self.lqr.gamma*torch.mean(torch.matmul(XU_prime, XU_prime.transpose(2, 3)), 1)

        A = torch.cat((torch.reshape(A, (A.shape[0], (self.lqr.N_x+self.lqr.N_u)**2, )),\
                      (1-self.lqr.gamma)*torch.ones((A.shape[0], 1), dtype=type)), -1)

        b = L.flatten()

        c = -torch.cat((torch.reshape(torch.diag_embed(self.Sigma),\
                      ((self.lqr.N_x+self.lqr.N_u)**2, 1)),\
                       torch.tensor([[1.]], dtype=type)), 0)

        c = c.flatten()

        x_star, obj_fun, time = self.__call_solver__(A, b, c)

        if x_star is not None:
            S = torch.tensor(x_star, dtype=type).unsqueeze(1)

            return A, b, c, torch.reshape(S[:-1, :], (self.lqr.N_u + self.lqr.N_x,\
                   self.lqr.N_u + self.lqr.N_x)), S[-1, :], None, None, obj_fun, x_star, time
        else:
            return A, b, c, None, None, None, None, obj_fun, x_star, time

    def policy_evaluation_symm(self, X, U, L, Xplus, Uprime):

        XU = torch.cat((X, U), 1)

        XU_prime = torch.cat((Xplus, Uprime), 2)

        A = torch.matmul(XU, XU.transpose(1,2)) -\
            self.lqr.gamma*torch.mean(torch.matmul(XU_prime, XU_prime.transpose(2, 3)), 1)

        mask = torch.cat(A.shape[0]*[torch.eye(A.shape[1], A.shape[2], dtype=torch.bool).unsqueeze(0)], 0)
        A = A + A.clone().masked_fill_(mask, 0)

        #extract here the upper triangular part
        A = A[:,torch.triu(torch.ones(A.shape[1], A.shape[2]))==1]

        A = torch.cat((A, (1-self.lqr.gamma)*torch.ones((A.shape[0], 1), dtype=type)), -1)

        b = L.flatten()

        Sigma_diag = torch.diag_embed(self.Sigma)

        c = -torch.cat((Sigma_diag[torch.triu(torch.ones(Sigma_diag.shape[0], Sigma_diag.shape[1]))==1],\
                       torch.tensor([1.], dtype=type)), 0)

        x_star, obj_fun, time = self.__call_solver__(A, b, c)

        if x_star is not None:

            S = torch.tensor(x_star, dtype=type)

            Q = torch.zeros(self.lqr.N_u + self.lqr.N_x, self.lqr.N_u + self.lqr.N_x, dtype=type)

            Q[torch.triu(torch.ones(self.lqr.N_u + self.lqr.N_x, self.lqr.N_u + self.lqr.N_x, dtype=type))==1] = S[:-1]
            mask = torch.eye(self.lqr.N_u + self.lqr.N_x, self.lqr.N_u + self.lqr.N_x, dtype=torch.bool)
            Q = Q + Q.clone().transpose(0,1).masked_fill_(mask, 0)

            return A, b, c, Q, S[-1], None, None, obj_fun, x_star, time
        else:
            return A, b, c, None, None, None, None, obj_fun, x_star, time
