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
            #torch.normal --> takes as input the std deviation!
            W = torch.reshape(self.normal.sample((X.shape[0]*K, )), (X.shape[0], K, self.N_x, 1))
            #W = torch.normal(0, self.sigma, size=(X.shape[0], K, self.N_x, 1), dtype=type)

        L_x = torch.matmul(torch.matmul(X.transpose(1,2), self.Q), X)
        L_u = torch.matmul(torch.matmul(U.transpose(1,2), self.R), U)

        return torch.cat(K*[X_plus.unsqueeze(1)], 1) + W, L_x + L_u, W

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
