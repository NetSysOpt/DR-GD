import numpy as np
import pickle
import torch
import gzip
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import SimpleProblem

from scipy import io as sio
from scipy.sparse import csc_matrix
from scipy.io import loadmat
import scs
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

seed = 17
num_examples_train = 440
num_examples_test = 100
valid_num = 0
test_num = 0
factor = 0.1

instance_num = 3913
data = loadmat(f'/QPLIB/QPLIB_{instance_num}.mat')
save_dir = "/datasets/qplib"
folder_name_train = "random_qplib_dataset_instance_{}_{}_ex{}".format(instance_num, factor, num_examples_train)
folder_name_test = "random_qplib_dataset_instance_{}_{}_ex{}".format(instance_num, factor, num_examples_test)

if not os.path.exists(os.path.join(save_dir, folder_name_train)):
    os.makedirs(os.path.join(save_dir, folder_name_train))
    
if not os.path.exists(os.path.join(save_dir, folder_name_test)):
    os.makedirs(os.path.join(save_dir, folder_name_test))

P_ini = data['P'] * 2
c_ini = data['c'].flatten()
A_ini = data['A']
b_ini = data['b'].flatten()
G_ini = data['G'] 
h_ini = data['h'].flatten()
l = data['lb'].flatten()
u = data['ub'].flatten()

num_var = P_ini.shape[0]
num_ineq = G_ini.shape[0]
num_eq = A_ini.shape[0]

print(instance_num, num_var, num_ineq, num_eq)

def perturb(P, c, A, b, G, h):
    #perturb quadratic matrix
    chosed_var = np.arange(num_var)
    chosed_ineq = np.arange(num_ineq)
    chosed_eq = np.arange(num_eq)
    
    ini_q_matrix = P.copy()
    ini_p_vec = c.copy()
    ini_ineq_matrix = G.copy()
    ini_ineq_rhs = h.copy()
    ini_eq_matrix = A.copy()
    ini_eq_rhs = b.copy()
    
    q_matrix = ini_q_matrix[chosed_var, :]
    q_matrix = q_matrix[:, chosed_var]    
    #perturb coefficient matrix
    nonzero_indices = np.nonzero(ini_q_matrix)
    nonzero_values = ini_q_matrix[nonzero_indices]
    perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
    q_matrix[nonzero_indices] = perturbed_values

    #perturb coefficient vector
    p_vec = ini_p_vec.copy()
    nonzero_indices = np.nonzero(ini_p_vec)
    nonzero_values = ini_p_vec[nonzero_indices]
    perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
    p_vec[nonzero_indices] = perturbed_values
    p_vec = p_vec[chosed_var]

    if G.shape[0] > 0:
        #perturb coefficient vector
        ineq_matrix = ini_ineq_matrix.copy()
        nonzero_indices = np.nonzero(ini_ineq_matrix)
        nonzero_values = ini_ineq_matrix[nonzero_indices]
        perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
        ineq_matrix[nonzero_indices] = perturbed_values
        ineq_matrix = ineq_matrix[chosed_ineq, :]
        ineq_matrix = ineq_matrix[:, chosed_var]
    else:
        ineq_matrix = G

    if h.shape[0] > 0:
        #perturb coefficient vector
        ineq_rhs = ini_ineq_rhs.copy()
        nonzero_indices = np.nonzero(ini_ineq_rhs)
        nonzero_values = ini_ineq_rhs[nonzero_indices]
        perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
        ineq_rhs[nonzero_indices] = perturbed_values
        ineq_rhs = ineq_rhs[chosed_ineq]
    else:
        ineq_rhs = h

    if A.shape[0] > 0:
        #perturb coefficient vector
        eq_matrix = ini_eq_matrix.copy()
        nonzero_indices = np.nonzero(ini_eq_matrix)
        nonzero_values = ini_eq_matrix[nonzero_indices]
        perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
        eq_matrix[nonzero_indices] = perturbed_values
        eq_matrix = eq_matrix[chosed_eq, :]
        eq_matrix = eq_matrix[:, chosed_var]
    else:
        eq_matrix = A

    if b.shape[0] > 0:
        #perturb coefficient vector
        eq_rhs = ini_eq_rhs.copy()
        nonzero_indices = np.nonzero(ini_eq_rhs)
        nonzero_values = ini_eq_rhs[nonzero_indices]
        perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
        eq_rhs[nonzero_indices] = perturbed_values
        eq_rhs = eq_rhs[chosed_eq]
    else:
        eq_rhs = b
    
    return q_matrix, p_vec, eq_matrix, eq_rhs, ineq_matrix, ineq_rhs


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
    solver = scs.SCS(data, cone_dict, eps_abs=tol, eps_rel=tol, verbose=True, max_iters=1000000)
    
    sol = solver.solve()
    print(sol['info']['pobj'])
    
    if sol['info']['status'] in ["solved"]:#, "solved (inaccurate - reached max_iters)"]:
        return sol['x'], sol['y'], sol['s'], sol['info']['iter'], sol['info']['pobj']
    else:
        return None, None, None, None, None


P = []
c = []
A = []
b = []
G = []
h = []

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
while count < num_examples_train:
    PP, cc, AA, bb, GG, hh = perturb(P_ini, c_ini, A_ini, b_ini, G_ini, h_ini)
    x, y, s, iter, obj = solve_scs(PP, cc, AA, bb, GG, hh, l, u)
    
    if x is not None:
        data = {'P': PP, 'c': cc, 'A': AA, 'b': bb, 'G': GG, 'h': hh, 'l': l, 'u': u,
                'X': x, 'Y': y, 'S': s, 'iter': iter, 'obj': obj}

        # save the data as .gz file
        with gzip.open(os.path.join(save_dir, folder_name_train, "instance_{}.gz".format(count)), 'wb') as f:
            pickle.dump(data, f)
        count += 1
        print(count)

count = 0
while count < num_examples_test:
    PP, cc, AA, bb, GG, hh = perturb(P_ini, c_ini, A_ini, b_ini, G_ini, h_ini)
    x, y, s, iter, obj = solve_scs(PP, cc, AA, bb, GG, hh, l, u)
    
    if x is not None:
        data = {'P': PP, 'c': cc, 'A': AA, 'b': bb, 'G': GG, 'h': hh, 'l': l, 'u': u,
                'X': x, 'Y': y, 'S': s, 'iter': iter, 'obj': obj}

        # save the data as .gz file
        with gzip.open(os.path.join(save_dir, folder_name_test, "instance_{}.gz".format(count)), 'wb') as f:
            pickle.dump(data, f)
        count += 1
        print(count)