import numpy as np
import torch


def dip(X, Z):
    """ 
    X: training data
    Z: DA data
    """
    n, d = X.size()
    beta = torch.zeros(d, requires_grad=True)
    gamma = 0
    mu = 0.001
    iter = 100
    optimizer = torch.optim.adam(beta)

    for iter in range(iter):
        loss = torch.mean(torch.square(X-torch.dot(X, beta)))
        violation = torch.mean(torch.dot(X, beta) - torch.dot(Z, beta))

        # compute augmented langrangian
        lagrangian = loss + gamma * violation
        augmentation = violation ** 2
        aug_lagrangian = lagrangian + 0.5 * mu * augmentation

        # optimization step on augmented lagrangian
        optimizer.zero_grad()
        aug_lagrangian.backward()
        optimizer.step()

        # compute delta for gamma
        if iter >= 2 * opt.stop_crit_win and iter % (2 * opt.stop_crit_win) == 0:
            t0, t_half, t1 = aug_lagrangians_val[-3][1], aug_lagrangians_val[-2][1], aug_lagrangians_val[-1][1]

            # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
            if not (min(t0, t1) < t_half < max(t0, t1)):
                delta_gamma = -np.inf
            else:
                delta_gamma = (t1 - t0) / opt.stop_crit_win
        else:
            delta_gamma = -np.inf  # do not update gamma nor mu




if __name__ == "__main__":
    # generate some data
    # do function call
    pass
