import numpy as np
import scipy.linalg
import torch

type = torch.float64

class dlqr:

    def __init__(self, A, B, C, rho, gamma, sigma=0):

        self.N_x = B.shape[0]
        self.N_u = B.shape[1]

        self.A = A
        self.B = B
        self.Q = torch.matmul(torch.transpose(C, 0, 1), C)
        self.R = rho*torch.eye(self.N_u, dtype=type)

        self.sigma = sigma
        self.gamma = gamma
        self.normal = torch.distributions.normal.Normal(torch.zeros((self.N_x, 1), dtype=type),\
                      self.sigma*torch.ones((self.N_x, 1), dtype=type))

    def simulate(self, K, X, U):

        # x_{k+1} = A*x_{k} + B*u_{k} + w_{k}, w_{k}~Gaussian(0, sigma^2*I)
        X_plus = torch.matmul(self.A, X) + torch.matmul(self.B, U)

        if self.sigma == 0:

            W = torch.zeros((X.shape[0], K, self.N_x, 1), dtype=type)

        else:
            #TODO is there a more efficient way to do that?? I do not think so...
            #sampling is really expensive
            #torch.normal --> takes as input the std deviation!
            W = torch.reshape(self.normal.sample((X.shape[0]*K, )), (X.shape[0], K, self.N_x, 1))
            #W = torch.normal(0, self.sigma, size=(X.shape[0], K, self.N_x, 1), dtype=type)

        L_x = torch.matmul(torch.matmul(X.transpose(1,2), self.Q), X)
        L_u = torch.matmul(torch.matmul(U.transpose(1,2), self.R), U)

        return X_plus.unsqueeze(1) + W, L_x + L_u, W

    def optimal_solution(self):

        P = scipy.linalg.solve_discrete_are(np.sqrt(self.gamma)*self.A.numpy(), self.B.numpy(),
                                            self.Q.numpy(), self.R.numpy()/self.gamma)

        K = -self.gamma*np.dot(np.dot(np.dot(scipy.linalg.inv(self.R.numpy() +
             self.gamma*np.dot(np.dot(np.transpose(self.B.numpy()), P), self.B.numpy())),
             np.transpose(self.B.numpy())), P), self.A.numpy())

        q = self.sigma**2*(self.gamma)/(1-self.gamma)*P.trace()

        return torch.tensor(P), torch.tensor(K), torch.tensor(q)

    def optimal_q(self, P, e, c):

        Qxx_star = self.Q + self.gamma*torch.matmul(torch.matmul(torch.transpose(\
                   self.A, 1, 0), P), self.A)
        Quu_star = self.R + self.gamma*torch.matmul(torch.matmul(torch.transpose(\
                   self.B, 1, 0), P), self.B)
        Qxu_star = self.gamma*torch.matmul(torch.matmul(torch.transpose(self.A, 1, 0), \
                   P), self.B)

        Qstar = torch.cat((torch.cat((Qxx_star, Qxu_star), 1), \
                torch.cat((torch.transpose(Qxu_star, 1, 0), Quu_star), 1)), 0)

        E_Qstar = (torch.diag_embed(c)*Qstar).sum()+e

        gap = (self.gamma)/(1-self.gamma)*\
              (torch.matmul(torch.matmul(torch.matmul(Qxu_star, torch.inverse(Quu_star)),\
               torch.transpose(Qxu_star, 0, 1)),self.sigma**2*torch.eye(self.N_x, dtype=type))).sum()

        return Qstar, E_Qstar.item(), gap

class cart_pole:

    def __init__(self, m_c, m_p, l, delta_t, C, rho, gamma, sigma):

        #system with 4 states and 1 input
        self.N_u = 1
        self.N_x = 4

        self.m_c = m_c
        self.m_p = m_p
        self.l = l

        self.g = 9.8

        self.delta_t = delta_t

        self.sigma = sigma
        self.gamma = gamma
        self.Q = torch.matmul(torch.transpose(C, 0, 1), C)
        self.R = rho*torch.eye(self.N_u, dtype=type)
        #this is the distribution of noise affecting the dynamics of the system
        self.normal = torch.distributions.normal.Normal(torch.zeros((self.N_x, 1), dtype=type),\
                      self.sigma*torch.ones((self.N_x, 1), dtype=type))

    def linearized_system(self):

        raise Exception('not implemented')

    def __f1__(self, X2, X3, U):

        return (self.l*self.__f2__(X2, X3, U)-self.g*torch.sin(X2))/torch.cos(X2)

    def __f2__(self, X2, X3, U):

        C = torch.cos(X2)/((self.m_p+self.m_c)*self.l - self.m_p*self.l*torch.cos(X2)**2)

        return C*(U + (self.m_p + self.m_c)*self.g*torch.sin(X2)/(torch.cos(X2)) - self.m_p*self.l*X3**2*torch.sin(X2))

    def __split(self, X):

        return X[:, 0, :].unsqueeze(1), X[:, 1, :].unsqueeze(1), X[:, 2, :].unsqueeze(1), X[:, 3, :].unsqueeze(1)

    def __f__(self, X, U):

        '''
        forward Euler discretization
        '''
        X0, X1, X2, X3 = self.__split(X)

        X0_new = X0 + self.delta_t*X1

        X1_new = X1 + self.delta_t*self.__f1__(X2, X3, U)

        X2_new = X2 + self.delta_t*X3

        X3_new = X3 + self.delta_t*self.__f2__(X2, X3, U)

        X_plus = torch.cat((X0_new, X1_new, X2_new, X3_new), 1)

        return X_plus

    def simulate(self, K, X, U):
        '''
        see equations (23)-(24) in https://coneural.org/florian/papers/05_cart_pole.pdf for a model
        '''

        X_plus = self.__f__(X, U)

        if self.sigma == 0:

            W = torch.zeros((X.shape[0], K, self.N_x, 1), dtype=type)

        else:
            #TODO is there a more efficient way to do that?? I do not think so...
            #sampling is really expensive
            #torch.normal --> takes as input the std deviation!
            W = torch.reshape(self.normal.sample((X.shape[0]*K, )), (X.shape[0], K, self.N_x, 1))
            #W = torch.normal(0, self.sigma, size=(X.shape[0], K, self.N_x, 1), dtype=type)

        L_x = torch.matmul(torch.matmul(X.transpose(1,2), self.Q), X)
        L_u = torch.matmul(torch.matmul(U.transpose(1,2), self.R), U)

        return X_plus.unsqueeze(1) + W, L_x + L_u, W

class pendulum:

    def __init__(self, m, l, k, delta_t, C, rho, gamma, sigma):

        #system with 4 states and 1 input
        self.N_u = 1
        self.N_x = 2

        self.m = m
        self.l = l
        self.k = k

        self.g = 9.8

        self.delta_t = delta_t

        self.sigma = sigma
        self.gamma = gamma
        self.Q = torch.matmul(torch.transpose(C, 0, 1), C)
        self.R = rho*torch.eye(self.N_u, dtype=type)
        #this is the distribution of noise affecting the system
        self.normal = torch.distributions.normal.Normal(torch.zeros((self.N_x, 1), dtype=type),\
                      self.sigma*torch.ones((self.N_x, 1), dtype=type))

    def linearized_system(self):

        raise Exception('not implemented')

    def __f1__(self, X0, X1, U):

        return -(self.g/self.l)*torch.sin(X0)-(self.k/(self.m*self.l))*X1 + U

    def __split(self, X):

        return X[:, 0, :].unsqueeze(1), X[:, 1, :].unsqueeze(1)

    def __f__(self, X, U):

        '''
        forward Euler discretization
        '''
        X0, X1 = self.__split(X)

        X0_new = X0 + self.delta_t*X1

        X1_new = X1 + self.delta_t*self.__f1__(X0, X1, U)

        X_plus = torch.cat((X0_new, X1_new), 1)

        return X_plus

    def simulate(self, K, X, U):
        '''
        see equations (23)-(24) in https://coneural.org/florian/papers/05_cart_pole.pdf for a model
        '''

        X_plus = self.__f__(X, U)

        if self.sigma == 0:

            W = torch.zeros((X.shape[0], K, self.N_x, 1), dtype=type)

        else:
            #TODO is there a more efficient way to do that?? I do not think so...
            #sampling is really expensive
            #torch.normal --> takes as input the std deviation!
            W = torch.reshape(self.normal.sample((X.shape[0]*K, )), (X.shape[0], K, self.N_x, 1))
            #W = torch.normal(0, self.sigma, size=(X.shape[0], K, self.N_x, 1), dtype=type)

        L_x = torch.matmul(torch.matmul(X.transpose(1,2), self.Q), X)
        L_u = torch.matmul(torch.matmul(U.transpose(1,2), self.R), U)

        return X_plus.unsqueeze(1) + W, L_x + L_u, W
