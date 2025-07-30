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

from utils_l2ws import *
import default_args

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
tol = 1e-4
max_iters = 100000
dr_gd = False

result_dir = '/result'
runId_dict = {    
    'simple_rhs_200': result_dir,
    'simple_rhs_500': result_dir,
    'simple_rhs_1000': result_dir,
}

def main():
    parser = argparse.ArgumentParser(description='SCS_unroll')
    parser.add_argument('--probType', type=str, default='emnist', #default='qplib_8616',
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
    parser.add_argument('--etaBase', type=float,
        help='base learning rate for the neural network')
    parser.add_argument('--timeMode', type=str, default='cpu')
    parser.add_argument('--supervised', type=bool)
    parser.add_argument('--train_unrolls', type=int)

    prefix = ""
    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.l2ws_default_args(args['probType'])
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
    
    args['runId'] = runId_dict[args['probType']]

    test_net(filepath, args)


def load_data(filepath, index):
    P, c, A, b, G, h, X, Y, S, ITER, OBJ_VAL, W, THETA = [],[],[],[],[],[],[],[],[],[],[],[],[]
    for id in index:
        instance_name = "instance_{}.gz".format(id)
        # instance_name = "qplib8845_{}.gz".format(id)
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
            W.append(data_tmp['W'])
            THETA.append(data_tmp['theta'])
    l = data_tmp['l']
    u = data_tmp['u']
    if l is not None:
        l = np.tile(l, (len(index), 1))
    if u is not None:
        u = np.tile(u, (len(index), 1))
    
    P, c, A, b, G, h, X, Y, S, ITER, OBJ_VAL = np.array(P), np.array(c), np.array(A), np.array(b), np.array(G), np.array(h), np.array(X), np.array(Y), np.array(S), np.array(ITER), np.array(OBJ_VAL)
    THETA = np.array(THETA)
    W = np.array(W)
    data = {'P': P, 'c': c, 'A': A, 'b': b, 'G': G, 'h': h, 'l': l, 'u': u, 
        'X': X, 'Y': Y, 'S': S, 'iter': ITER, 'obj': OBJ_VAL,
        'W': W, 'theta': THETA}
    data = SimpleProblem(data, valid_num=0, test_num=0)
    
    return data

def DR_gd_torch(M, q, n, m, max_iter=10000, tol=1e-4, w=None, u_tilde=None):
    if w is None:
        w = torch.zeros(M.shape[0], device=DEVICE).unsqueeze(-1)
    w_prev = w     
    if u_tilde is None:
        u_tilde= torch.zeros(M.shape[0], device=DEVICE).unsqueeze(-1)

    for i in range(max_iter):        
        b = w - q
        
        for _ in range(1):
            res = u_tilde + M @ u_tilde - b
            grad = res + M.T @ res
            AAT_res = grad + M @ grad
            eta = torch.sum(res * AAT_res) / torch.sum(AAT_res * AAT_res) 
            u_tilde = u_tilde - eta * grad
        
        u = 2 * u_tilde - w
        u[n+m:] = torch.clamp(u[n+m:], min=0)
        w = w + 1.5 * (u - u_tilde)
            
        if i % 1 == 0:
            error = torch.linalg.norm(w - w_prev)
            if error < tol:
                break

        w_prev = w.clone()
            
    return u[:n], u[n:], i

def DR_torch(M, q, n, m, max_iter=100000000, tol=1e-4, w=None,):
    if w is None:
        w = torch.zeros((M.shape[1],1), device=DEVICE)
    else:
        w = w.clone()
    w_prev = w
    M_tmp = torch.linalg.inv(torch.eye(M.shape[0], device=DEVICE) + M)
    
    for i in range(max_iter):
        u_tilde = M_tmp @ (w - q)
        u = 2 * u_tilde - w
        u[n+m:] = torch.clamp(u[n+m:], 0)
        w = w + 1.5*(u - u_tilde)
        
        error = torch.linalg.norm(w - w_prev)
        if error < tol:
            break
        
        w_prev = w
    
    return u[:n], w, i

import torch

def root(mu, eta, p, r):
    a = 1 + torch.sum(r * r, dim=1)
    b = torch.sum(r * mu, dim=1) - 2 * torch.sum(r * p, dim=1) - eta
    c = torch.sum(p * (p - mu), dim=1)
    
    return (-b + torch.sqrt(b**2 - 4 * a * c)) / (2 * a)

def DR_homo_torch(M, q, n, m, max_iter=100000, tol=1e-4, mu=None, eta=None):
    device = M.device  
    batch_size = M.shape[0]  

    if mu is None:
        mu = torch.zeros((batch_size, M.shape[1], 1),device=device)
    if eta is None:
        eta = torch.ones((batch_size,1), device=device)
    
    mu_prev = mu.clone()
    eta_prev = eta.clone()
    
    I = torch.eye(M.shape[1], device=device).unsqueeze(0).expand_as(M)
    r = torch.linalg.solve(I + M, q)
    
    for i in range(max_iter):
        # Calculate `u_tilde`
        p = torch.linalg.solve(I + M, mu)
        tau_tilde = root(mu, eta, p, r)
        z_tilde = p - r * tau_tilde.unsqueeze(-1)
        
        # Calculate `u`
        z = 2 * z_tilde - mu
        z[:, n+m:] = torch.maximum(z[:, n+m:], torch.zeros_like(z[:, n+m:]))
        tau = torch.maximum(2 * tau_tilde - eta, torch.zeros_like(eta))
        
        mu_prev = mu.clone()
        eta_prev = eta.clone()
        # Update `w`
        mu = mu + 1.5*(z - z_tilde)
        eta = eta + 1.5*(tau - tau_tilde)

        # Calculate error
        error = torch.linalg.norm(mu/eta - mu_prev/eta_prev, dim=1)

        if error.item() < tol:
            print("DR Splitting for homogeneous embedding stopped at iteration: ", (i + 1))
            break
    
    if i == max_iter - 1:
        print("DR Splitting stopped at maximum iteration: ", max_iter)
    
    return z[:, :n] / tau, z[:, n:] / tau, i


def test_net(filepath, args):
    print("time mode: ", args['timeMode'])
    print("fixed point tolerance: ", tol)
    print("max iterations: ", max_iters)
    
    num_examples = args['simpleEx']
    frac_valid = 1/12
    num_train = int(num_examples*(1-2*frac_valid))
    num_valid = int(num_examples*frac_valid)

    test_indices = np.arange(num_examples)
    print("number of test samples: ", len(test_indices))

    if 'rhs' in args['probType']:
        input_size = args['simpleEq']
        output_size = args['simpleVar'] + args['simpleEq'] + args['simpleIneq']
    elif 'diag' in args['probType']:
        input_size = args['simpleVar'] + args['simpleEq']
        output_size = args['simpleVar'] + args['simpleEq'] + args['simpleIneq']
    elif 'emnist' in args['probType']:
        input_size = args['simpleVar']
        output_size = args['simpleVar'] * 3
    hidden_layers = args['hiddenSize']
    solver_net = NN_l2ws(input_size, output_size, hidden_layers)
    # directory of the checkpoints
    checkpoint_dir = args['runId']
    if args['supervised']:
        checkpoint_file = 'checkpoint_' + args['probType'] + '_' + str(args['train_unrolls']) + '_reg' + '.pth'
    else:
        checkpoint_file = 'checkpoint_' + args['probType'] + '_' + str(args['train_unrolls']) + '_fp' + '.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    print("loading checkpoint from: ", checkpoint_path)
    solver_net.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    solver_net.to(DEVICE)
    solver_net.eval()
    print("file path: ", filepath)
    
    obj_pred = []
    primal_dist = []
    dual_dist = []
    eq_dist_pred = []
    ineq_dist_pred = []
    time_pred = []
    iter_dr_gd_ws = []
    iter_dr_ws = []
    iter_dr_gd_orig = []
    iter_dr_orig = []
    time_dr_pred = []
    time_dr_ws = []
    time_dr_orig = []


    for idx in test_indices:
        data = load_data(filepath, [idx])
        q_train = data.q.to(DEVICE)
        M_train = data.M.to(DEVICE)
        theta = data.theta.to(DEVICE)
    
        if idx == test_indices[0]:
            with torch.no_grad():
                w = solver_net(theta)
        
        if args['timeMode'] == 'cpu':
            start_time = time.time()
            with torch.no_grad():
                w = solver_net(theta)
                w, w_prev, eta, eta_prev = data.fixed_point_homo(w.unsqueeze(-1), M_train, q_train, k=args['train_unrolls'])
            end_time = time.time()
            time_pred.append(end_time-start_time)

        ###############################################################
        P = data.P[[0]].detach().cpu().numpy()[0] 
        c = data.c[[0]].detach().cpu().numpy().flatten()
        A = data.my_A[[0]].detach().cpu().numpy()[0] 
        b = data.my_b[[0]].detach().cpu().numpy().flatten()        
        
        
        w = w / eta
        x_ws = w[:, :data.n_var].detach().cpu().numpy().flatten() 
        y_ws = w[:, data.n_var:].detach().cpu().numpy().flatten() 
        s_ws = np.zeros_like(y_ws)

        
        cone_dict = {'z': data.num_zero_cone, 'l': data.num_linear_cone}
        scs_data = {'P': csc_matrix(P), 'c': c, 'A': csc_matrix(A), 'b': b, 'cone': cone_dict}
    
        # warm start
        # solver_ws = scs.SCS(scs_data, cone_dict, verbose=False, normalize=use_normalize, scale=scale, acceleration_lookback=acceleration_lookback, use_indirect=use_indirect, adaptive_scale=adaptive_scale)
        solver_ws = scs.SCS(scs_data, cone_dict, eps_abs=1e-4, eps_rel=1e-4, verbose=False, 
                                    acceleration_lookback=0, normalize=False, adaptive_scale=False,
                                    rho_x=1.0, scale=1.0, alpha=1.)
        result_ws = solver_ws.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
        iter = result_ws['info']['iter'] #+ args['train_unrolls']

        iter_dr_ws.append(iter)
        time_dr_ws.append(1e-3*(result_ws['info']['solve_time']+result_ws['info']['setup_time']))
        
    
    for idx in test_indices:
        data = load_data(filepath, [idx])
        q_train = data.q.to(DEVICE)
        M_train = data.M.to(DEVICE)
        if dr_gd:
            with torch.no_grad():
                _, _, iter = DR_gd_torch(M_train.squeeze(0), q_train.squeeze(0), data.n_var, data.n_eq, max_iter=max_iters, tol=tol,)
            iter_dr_gd_orig.append(iter+1)

        P = data.P[[0]].detach().cpu().numpy()[0] 
        c = data.c[[0]].detach().cpu().numpy().flatten()
        A = data.my_A[[0]].detach().cpu().numpy()[0] 
        b = data.my_b[[0]].detach().cpu().numpy().flatten()    

        cone_dict = {'z': data.num_zero_cone, 'l': data.num_linear_cone}
        scs_data = {'P': csc_matrix(P), 'c': c, 'A': csc_matrix(A), 'b': b, 'cone': cone_dict}
        solver_orig = scs.SCS(scs_data, cone_dict, verbose=False, normalize=False,
                              scale=1.0, rho_x=1.0, alpha=1., 
                              acceleration_lookback=0, adaptive_scale=False)
        result_orig = solver_orig.solve()
        
        iter_dr_orig.append(result_orig['info']['iter'])
        time_dr_orig.append(1e-3*(result_orig['info']['solve_time']+result_orig['info']['setup_time']))
        
        
    if dr_gd:
        df = pd.DataFrame({'instance': test_indices, 
                       'iter_dr_orig': iter_dr_orig, 'iter_dr_gd_orig': iter_dr_gd_orig,
                       'iter_dr_ws': iter_dr_ws, 'iter_dr_gd_ws': iter_dr_gd_ws, 
                    })

        df['iter_dr_ratio'] = (df['iter_dr_orig'] - df['iter_dr_ws'])/df['iter_dr_orig']
        df['iter_dr_gd_ratio'] = (df['iter_dr_gd_orig'] - df['iter_dr_gd_ws'])/df['iter_dr_gd_orig']
        df.to_csv(os.path.join(args['runId'], args['probType']+ "_" + args['timeMode'] + ".csv"), index=False)

        print('-------------------   DR   -----------------')
        print("Avg.Iter Orig: {:.3f}|\t Avg.Iter Ws: {:.3f}|\t Ratio: {:.3f}".format(np.mean(iter_dr_orig), np.mean(iter_dr_ws), np.mean(df['iter_dr_ratio'])))

        print('-------------------   DR_GD   -----------------')
        print("Avg.Iter Orig: {:.3f}|\t Avg.Iter Ws: {:.3f}|\t Ratio: {:.3f}".format(np.mean(iter_dr_gd_orig), np.mean(iter_dr_gd_ws), np.mean(df['iter_dr_gd_ratio'])))
    else:
        df = pd.DataFrame({'instance': test_indices, 
                   'iter_dr_orig': iter_dr_orig, 
                   'iter_dr_ws': iter_dr_ws, 
                   'time_dr_orig': time_dr_orig,
                     'time_dr_ws': time_dr_ws
                })

        df['iter_dr_ratio'] = (df['iter_dr_orig'] - df['iter_dr_ws'])/df['iter_dr_orig']
        df['time_dr_ratio'] = (df['time_dr_orig'] - df['time_dr_ws'])/df['time_dr_orig']
        df['total_time'] = time_pred + df['time_dr_ws']
        df['total_time_ratio'] = (df['time_dr_orig'] - df['total_time'])/df['time_dr_orig']
        # df.to_csv(os.path.join(args['runId'], args['probType']+ "_" + args['timeMode'] + ".csv"), index=False)

        print('-------------------   DR   -----------------')
        print("Avg.Iter Orig: {:.3f}|\t Avg.Iter Ws: {:.3f}|\t Ratio: {:.3f}".format(np.mean(iter_dr_orig), np.mean(iter_dr_ws), np.mean(df['iter_dr_ratio'])))
        print('-------------------   Time   -----------------')
        print("Avg.Time Orig: {:.3f}|\t Avg.Time Ws: {:.3f}|\t Avg.Time Total: {:.3f}, Total Ratio: {:.3f}".format(np.mean(time_dr_orig), np.mean(time_dr_ws), np.mean(df['total_time']), np.mean(df['total_time_ratio'])))
if __name__=='__main__':
    main()