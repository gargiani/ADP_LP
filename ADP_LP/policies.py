import torch

type = torch.float64

def linear_policy(M, X, epsilon=None):

    U = torch.matmul(M, X)

    if epsilon is not None:
        normal = torch.distributions.normal.Normal(torch.zeros((U.shape[-2], U.shape[-1]), dtype=type),\
                 torch.ones((U.shape[-2], U.shape[-1]), dtype=type))
        noise = epsilon*normal.sample((U.shape[0], ))

        if len(U.shape)==3:
            U = U + noise
        else:
            U = U + noise.unsqueeze(1)#torch.reshape(epsilon*normal.sample((U.shape[0]*U.shape[1], )), U.shape)

    return U
