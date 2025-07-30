import numpy as np
import pickle
import torch

import scipy.sparse as spa
from scipy.sparse import csc_matrix
import sys
import os
import gzip
import scs


sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import SimpleProblem
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_default_dtype(torch.float64)
# Construct the problem
#       minimize	x' D x + y' I y - (1/gamma) * mu' x
#       subject to  1' x = 1
#                   F' x = y
#                   0 <= x <= 1
#
#formulate the problem
#       minimize [x' y'][D 0; 0 I][x; y] - (1/gamma) * [mu', 0] [x;y]
#       subject to  [1' 0][x;y] = [1; 0]
#                   [F' -I][x;y] = 0
#                   [0; -inf] <= [x;y] <= [1; inf]


seed = 17
np.random.seed(seed)
save_dir = "/datasets/simple"
num_examples_train = 440
num_examples_test = 100


k = 100      # Number of factors
n = 10 * k     # Number of assets, if not specified, n = k * 100
gamma = 1.0


num_var = n + k
num_ineq = 0
num_eq = k + 1
folder_name_train = "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples_train)
folder_name_test = "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples_test)
if not os.path.exists(os.path.join(save_dir, folder_name_train)):
    os.makedirs(os.path.join(save_dir, folder_name_train))
if not os.path.exists(os.path.join(save_dir, folder_name_test)):
    os.makedirs(os.path.join(save_dir, folder_name_test))


def generate_portfolio(n, k, gamma):
    F = spa.random(n, k, density=0.5, data_rvs=np.random.randn, format='csc')       # F_ij ~ N(0, 1), 50% nonzero elements
    D = spa.diags(np.random.rand(n) * np.sqrt(k), format='csc')     # D_ii ~ U(0, sqrt(k)), diagonal matrix
    mu = np.random.randn(n)     # mu_i ~ N(0, 1)

    P = spa.block_diag((2 * D, 2 * spa.eye(k)), format='csc')
    c = np.append(-mu / gamma, np.zeros(k))
    A = spa.vstack([spa.hstack([spa.csc_matrix(np.ones((1, n))),
                               spa.csc_matrix((1, k))]),
                    spa.hstack([F.T, -spa.eye(k)])
                ]).tocsc()
    ###################################
    b = np.hstack([1., np.zeros(k)])
    l = np.hstack([np.zeros(n), -np.inf * np.ones(k)])
    u = np.hstack([np.ones(n), np.inf * np.ones(k)])
    
    G = np.array([])
    h = np.array([])
    
    return P.toarray(), c, A.toarray(), b, G, h, l, u

def solve_scs(P, c, A, b, G, h, lb=None, ub=None, verbose=False, tol=1e-4):
    if A.shape[0] == 0 and b.shape[0] == 0:
        my_A = np.zeros((1, P.shape[0]))
        my_b = np.zeros((1, ))
    else:
        my_A = A
        my_b = b
        
    if G.shape[0] and h.shape[0]:
        my_A = np.vstack([my_A, G])
        my_b = np.hstack([my_b, h])
    
    if lb is not None:
        lb_idx = np.arange(len(lb))[lb != -np.inf]
        if len(lb_idx) > 0:
            A_lb = -np.eye(P.shape[0])[lb_idx, :]
            b_lb = -lb[lb_idx]
            my_A = np.vstack([my_A, A_lb])
            my_b = np.hstack([my_b, b_lb])
    
    if ub is not None:
        ub_idx = np.arange(P.shape[0])[ub != np.inf]
        if len(ub_idx) > 0:
            A_ub = np.eye(P.shape[0])[ub_idx, :]
            b_ub = ub[ub_idx]
            my_A = np.vstack([my_A, A_ub])
            my_b = np.hstack([my_b, b_ub])
    
    if A.shape[0] == 0 and b.shape[0] == 0:
        my_A = my_A[1:]
        my_b = my_b[1:]
        
    cone_dict = {'z': A.shape[0], 'l': my_A.shape[0]-A.shape[0]}
    data = {'P': csc_matrix(P), 'c': c, 'A': csc_matrix(my_A), 'b': my_b, 'cone': cone_dict}
    # solver = scs.SCS(data, cone_dict, eps_abs=tol, eps_rel=tol, verbose=verbose, 
    #                 acceleration_lookback=0, normalize=False, adaptive_scale=False,
    #                 rho_x=1.0, scale=1.0, alpha=1.0)
    solver = scs.SCS(data, cone_dict, eps_abs=tol, eps_rel=tol, verbose=False, normalize=True)
    
    sol = solver.solve()
    print(sol['info']['pobj'])
    
    if sol['info']['status'] == "solved":
        return sol['x'], sol['y'], sol['s'], sol['info']['iter'], sol['info']['pobj']
    else:
        return None, None, None, None, None
    
count = 0
while count < num_examples_train:
    PP, cc, AA, bb, GG, hh, l, u = generate_portfolio(n, k, gamma)
    x, y, s, iters, obj = solve_scs(PP, cc, AA, bb, GG, hh, l, u, verbose=False)
    if x is not None:
        data = {'P': PP, 'c': cc, 'A': AA, 'b': bb, 'G': GG, 'h': hh, 'l': l, 'u': u,
                'X': x, 'Y': y, 'S': s, 'iter': iters, 'obj': obj}  

        # save the data as .gz file
        with gzip.open(os.path.join(save_dir, folder_name_train, "instance_{}.gz".format(count)), 'wb') as f:
            pickle.dump(data, f)
        count += 1

count = 0
while count < num_examples_test:
    PP, cc, AA, bb, GG, hh, l, u = generate_portfolio(n, k, gamma)
    x, y, s, iters, obj = solve_scs(PP, cc, AA, bb, GG, hh, l, u, verbose=False)
    if x is not None:
        data = {'P': PP, 'c': cc, 'A': AA, 'b': bb, 'G': GG, 'h': hh, 'l': l, 'u': u,
                'X': x, 'Y': y, 'S': s, 'iter': iters, 'obj': obj}  

        # save the data as .gz file
        with gzip.open(os.path.join(save_dir, folder_name_test, "instance_{}.gz".format(count)), 'wb') as f:
            pickle.dump(data, f)
        count += 1
