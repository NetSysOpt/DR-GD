import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

import numpy as np
import pandas as pd
import pickle
import gzip
import time
import os
import argparse

from utils import *
import default_args
import osqp

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
solving the instances using osqp
'''

def main():
    parser = argparse.ArgumentParser(description='SCS_unroll')
    parser.add_argument('--probType', type=str, default='simple_rhs_100',
                        help='problem type')
    parser.add_argument('--simpleVar', type=int, 
        help='number of decision vars for simple problem')
    parser.add_argument('--simpleIneq', type=int,
        help='number of inequality constraints for simple problem')
    parser.add_argument('--simpleEq', type=int,
        help='number of equality constraints for simple problem')
    parser.add_argument('--simpleEx', type=int,
        help='total number of datapoints for simple problem')
    parser.add_argument('--epochs', type=int,
        help='number of neural network epochs')
    parser.add_argument('--batchSize', type=int,
        help='training batch size')
    parser.add_argument('--lr', type=float,
        help='neural network learning rate')
    parser.add_argument('--hiddenSize', type=int,
        help='hidden layer size for neural network')
    parser.add_argument('--earlyStop', type=int,
        help='number of epochs for early stopping')
    parser.add_argument('--embSize', type=int,
        help='embedding size')
    parser.add_argument('--numLayers', type=int,
        help='number of layers')
    parser.add_argument('--lambda1', type=float,
        help='scaling factor for the primal MSE loss')
    parser.add_argument('--etaBase', type=float, default=0.05,
        help='base learning rate for the neural network')
    parser.add_argument('--timeMode', type=str, default='cpu')

    prefix = ""
    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.method_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]
    print(args)

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        filepath = os.path.join(prefix + 'datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif 'simple_rhs' in prob_type:
        n_var = int(args['probType'].split('_')[2])
        args['simpleVar'] = n_var
        args['simpleIneq'] = n_var // 2
        args['simpleEq'] = n_var // 2
        filepath = os.path.join(prefix + 'datasets', 'simple_rhs', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif 'qplib' in prob_type:
        instance_num = "_".join(args['probType'].split('_')[1:])
        filepath = os.path.join(prefix + 'datasets', 'qplib', "random_qplib_dataset_instance_{}_ex{}".format(
            instance_num, args['simpleEx']))
    elif 'port' in prob_type:
        k = int(args['probType'].split('_')[1])
        n = 10 * k
        args['simpleVar'] = n + k
        args['simpleIneq'] = 0
        args['simpleEq'] = k + 1
        filepath = os.path.join(prefix + 'datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    else:
        raise NotImplementedError
    
    test_net(filepath, args)


def load_data(filepath, index):
    P, c, A, b, G, h, X, Y, S, ITER, OBJ_VAL = [],[],[],[],[],[],[],[],[],[],[]
    W = []
    for id in index:
        instance_name = "instance_{}.gz".format(id)
        instance_name = os.path.join(filepath, instance_name)
        with gzip.open(instance_name, 'rb') as f:
            data_tmp = pickle.load(f)
            P.append(data_tmp['P'])
            c.append(data_tmp['c'])
            A.append(data_tmp['A'])
            b.append(data_tmp['b'])
            G.append(data_tmp['G'])
            h.append(data_tmp['h'])
            X.append(data_tmp['X'])
            Y.append(data_tmp['Y'])
            S.append(data_tmp['S'])
            ITER.append(data_tmp['iter'])
            OBJ_VAL.append(data_tmp["obj"])
            
            # W.append(data_tmp['W'])
    l = data_tmp['l']
    u = data_tmp['u']
    if l is not None:
        l = np.tile(l, (len(index), 1))
    if u is not None:
        u = np.tile(u, (len(index), 1))
    
    P, c, A, b, G, h, X, Y, S, ITER, OBJ_VAL = np.array(P), np.array(c), np.array(A), np.array(b), np.array(G), np.array(h), np.array(X), np.array(Y), np.array(S), np.array(ITER), np.array(OBJ_VAL)
    data = {'P': P, 'c': c, 'A': A, 'b': b, 'G': G, 'h': h, 'l': l, 'u': u, 
        'X': X, 'Y': Y, 'S': S, 'iter': ITER, 'obj': OBJ_VAL}
    data = SimpleProblem(data, valid_num=0, test_num=0, calc_X=False)
    
    return data

def get_gap(x, y, P, c, b):
    if len(c.shape) == 1:
        c = c.reshape(-1, 1)
    if len(b.shape) == 1:
        b = b.reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    gap = np.abs(c.T @ x + x.T @ P @ x + b.T @ y) 
    scale = max(np.abs(c.T @ x), np.abs(x.T @ P @ x), np.abs(b.T @ y))
    
    return gap, scale


def test_net(filepath, args):
    use_normalize = True
    if 'qplib' in args['probType']:
        use_normalize = True
    use_indirect = False
    adaptive_scale = True
    eps_abs_target = 1e-4
    eps_rel_target = 1e-4
    
    print("probType: ", args['probType'])
    print("time mode: ", args['timeMode'])
    
    num_examples = args['simpleEx']

    test_indices = np.arange(num_examples)
    print("number of test samples: ", len(test_indices))

    iter_orig = []
    time_orig = []
    obj_true = []
    obj_osqp = []
    setup_time = []
    
    for idx in test_indices:
        data = load_data(filepath, [idx])
        P = data.P[[0]].detach().cpu().numpy()[0] 
        c = data.c[[0]].detach().cpu().numpy().flatten()
        A = data.my_A[[0]].detach().cpu().numpy()[0] 
        b = data.my_b[[0]].detach().cpu().numpy().flatten()
        l = np.hstack([b[:data.num_zero_cone], -np.inf * np.ones(data.num_linear_cone)])
        u = b
        
        gap = np.inf
        scale = 0
        eps_abs = 1e-4
        eps_rel = 1e-4
        count = 0
        while gap > eps_abs_target + eps_rel_target * scale and eps_abs > 1e-10 and eps_rel > 1e-8:
            problem = osqp.OSQP()
            problem.setup(P=csc_matrix(P), q=c, A=csc_matrix(A), l=l, u=u, verbose=False, max_iter=1000000, eps_abs=eps_abs, eps_rel=eps_rel, eps_prim_inf=1e-7, eps_dual_inf=1e-7)

            result = problem.solve()
            print(result.info.run_time)
            print(result.info.status)
            x = result.x
            y = result.y
            
            gap, scale = get_gap(x, y, P, c, b)
            # print(gap, scale)
            
            eps_abs /= 2
            eps_rel /= 2
            count += 1
        print(count)
            
        iter_orig.append(result.info.iter)
        time_orig.append(result.info.run_time)
        obj_true.append(data.OBJ_VAL.item())
        obj_osqp.append(result.info.obj_val)
        setup_time.append(result.info.setup_time)

    print('-------------------   OSQP    -----------------')
    print("Avg.Iter: {:.3f}|\t Avg.Time: {:.3f}|".format(np.mean(iter_orig), np.mean(time_orig)))
    print("OBJ True: {:.3f}|\t OBJ OSQP: {:.3f}".format(np.mean(obj_true), np.mean(obj_osqp)))
    print("Setup Time: {:.3f}".format(np.mean(setup_time)))
if __name__=='__main__':
    main()