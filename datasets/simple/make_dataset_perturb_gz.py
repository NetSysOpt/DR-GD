import numpy as np
import torch
import gzip
import pickle
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))

from scipy.sparse import csc_matrix
import scs
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

seed = 17
num_var = 200
num_ineq = 100
num_eq = 100
num_examples = 440
valid_num = 0
test_num = 0
factor = 0.1

# define the save_dir
save_dir = "/datasets/simple"
folder_name = "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples)

if not os.path.exists(os.path.join(save_dir, folder_name)):
    os.makedirs(os.path.join(save_dir, folder_name))


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
    solver = scs.SCS(data, cone_dict, verbose=verbose, max_iters=10000, use_indirect=False)
    
    sol = solver.solve()
    print(sol['info']['pobj'])
    
    if sol['info']['status'] == "solved":
        return sol['x'], sol['y'], sol['s'], sol['info']['iter'], sol['info']['pobj']
    else:
        return None, None, None, None, None


print(num_var, num_ineq, num_eq)
np.random.seed(seed)

def perturb(num_var, num_ineq, num_eq):

    P_ini = np.diag(np.logspace(-3, 1, num_var)*np.random.random(num_var))
    c_ini = np.random.random(num_var)
    A_ini = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    b_ini = np.random.uniform(-1, 1, size=(num_eq))
    G_ini = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    h_ini = np.sum(np.abs(G_ini@np.linalg.pinv(A_ini)), axis=1)
    
    return P_ini, c_ini, A_ini, b_ini, G_ini, h_ini

P = []
c = []
A = []
b = []
G = []
h = []
l = None
u = None

if l is not None and np.all(l == -np.inf):
    l = None
if u is not None and np.all(u == np.inf):
    u = None

X = []
Y = []
S = []
ITER = []
OBJ_VAL = []

count = 0
while count < num_examples:
    # PP, cc, AA, bb, GG, hh = perturb(P_ini, c_ini, A_ini, b_ini, G_ini, h_ini)
    PP, cc, AA, bb, GG, hh = perturb(num_var, num_ineq, num_eq)
    x, y, s, iter, obj = solve_scs(PP, cc, AA, bb, GG, hh, l, u, verbose=True)
    if x is not None:
        data = {'P': PP, 'c': cc, 'A': AA, 'b': bb, 'G': GG, 'h': hh, 'l': l, 'u': u,
                'X': x, 'Y': y, 'S': s, 'iter': iter, 'obj': obj}
        print(PP.max(), PP.min())
        print(x.min(), x.max())
        print(y.min(), y.max())
        # save the data as .gz file
        with gzip.open(os.path.join(save_dir, folder_name, "instance_{}.gz".format(count)), 'wb') as f:
            pickle.dump(data, f)
        count += 1
    print(count)