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
from model_utils import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

result_dir = '/result'
runId_dict = {
    'port_100': result_dir,
    'port_200': result_dir,
    'port_300': result_dir,
    'port_400': result_dir,
    
    'qplib_3913_0.1': result_dir,
    'qplib_4270_0.1': result_dir,
    'qplib_8845_0.1': result_dir,
    
    'simple': result_dir,
    
    'simple_rhs_200': result_dir,
    'simple_rhs_500': result_dir,
    'simple_rhs_1000': result_dir,
}

def main():
    parser = argparse.ArgumentParser(description='SCS_unroll')
    parser.add_argument('--probType', type=str, default='qplib_3547_0.1', 
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
    
    args['runId'] = runId_dict[args['probType']]

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
    scale = 0.1
    acceleration_lookback = 10
    use_indirect = False
    adaptive_scale = True
    scs_default = not 'simple' in args['probType']
    print("using scs default settings: ", scs_default)
    print("probType: ", args['probType'])
    print("acceleration_lookback: ", acceleration_lookback)
    print("use_normalize: ", use_normalize)
    print("scale: ", scale) 
    print("adaptive_scale: ", adaptive_scale)
    print("use_indirect: ", use_indirect)
    print("time mode: ", args['timeMode'])
    
    num_examples = args['simpleEx']

    test_indices = np.arange(num_examples)
    print("number of test samples: ", len(test_indices))

    solver_net = SCS_unroll(args['embSize'], args['numLayers'], args['etaBase'])
    if args['probType'] == 'simple':
        checkpoint_path = os.path.join(args['runId'], 'checkpoint_'+args['probType']+'_' + str(args['simpleVar']) + ".pth")
    else:
        checkpoint_path = os.path.join(args['runId'], 'checkpoint_'+args['probType']+".pth")
        
    print("loading checkpoint from: ", checkpoint_path)
    solver_net.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    solver_net.to(DEVICE)
    solver_net.eval()
    print("file path: ", filepath)
    
    obj_pred = []
    obj_true = []
    primal_dist = []
    dual_dist = []
    eq_dist_pred = []
    ineq_dist_pred = []
    time_pred = []
    iter_orig = []
    iter_ws = []
    time_orig = []
    time_ws = []
    scale_updates_orig = []
    scale_updates_ws = []
    time_setup = []
    
    acc_accept_ws = []
    acc_rej_ws = []
    
    acc_accept_orig = []
    acc_rej_orig = []

    for idx in test_indices:
        print("ws: ", idx)
        data = load_data(filepath, [idx])
        q_train = data.q.to(DEVICE)
        M_train = data.M.to(DEVICE)
    
        P = data.P[[0]].detach().cpu().numpy()[0] 
        c = data.c[[0]].detach().cpu().numpy().flatten()
        A = data.my_A[[0]].detach().cpu().numpy()[0] 
        b = data.my_b[[0]].detach().cpu().numpy().flatten()
    
        # do a warm start at the first run to making more stable timing
        if idx == test_indices[0]:
            with torch.no_grad():
                _, u, _, v = solver_net(q_train, M_train, data.n_var, data.n_eq) 
        
        if args['timeMode'] == 'cpu':
            start_time = time.time()
            with torch.no_grad():
                _, u, _, v = solver_net(q_train, M_train, data.n_var, data.n_eq)
            end_time = time.time()
            time_pred.append(end_time-start_time)
        elif args['timeMode'] == 'gpu':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                _, u, _, v = solver_net(q_train, M_train, data.n_var, data.n_eq)
            end_event.record()
            end_time = time.time()
            torch.cuda.synchronize()
            time_pred.append(start_event.elapsed_time(end_event)*1e-3)        
    
        X = u[:, :data.n_var].squeeze(-1)
        Y = u[:, data.n_var:].squeeze(-1)
    
        obj_pred.append(data.obj_fn([0], X).item())
        eq_dist_pred.append(data.eq_dist([0], X)[0].max().item())
        ineq_dist_pred.append(data.ineq_dist([0], X)[0].max().item())
    
        primal_dist.append(0)
        dual_dist.append(0)
    
        x_ws = X.detach().cpu().numpy().flatten()
        y_ws = Y.detach().cpu().numpy().flatten()
        s_ws = b - A @ x_ws
        
        cone_dict = {'z': data.num_zero_cone, 'l': data.num_linear_cone}
        scs_data = {'P': csc_matrix(P), 'c': c, 'A': csc_matrix(A), 'b': b, 'cone': cone_dict}
    
        # warm start
        if scs_default:
            solver_ws = scs.SCS(scs_data, cone_dict, verbose=False, normalize=use_normalize, scale=scale, acceleration_lookback=acceleration_lookback, use_indirect=use_indirect, adaptive_scale=adaptive_scale)
        else:
            solver_ws = scs.SCS(scs_data, cone_dict, verbose=False, normalize=False, 
                            scale=1.0, rho_x=1.0, alpha=1., 
                            acceleration_lookback=0, adaptive_scale=False)
        result_ws = solver_ws.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
        iter_ws.append(result_ws['info']['iter'])
        time_ws.append(1e-3*(result_ws['info']['solve_time'] + result_ws['info']['setup_time']))
        scale_updates_ws.append(result_ws['info']['scale_updates'])
        acc_accept_ws.append(result_ws['info']['accepted_accel_steps'])
        acc_rej_ws.append(result_ws['info']['rejected_accel_steps'])
        
    
    for idx in test_indices:
        print("orig: ", idx)
        data = load_data(filepath, [idx])
        P = data.P[[0]].detach().cpu().numpy()[0] 
        c = data.c[[0]].detach().cpu().numpy().flatten()
        A = data.my_A[[0]].detach().cpu().numpy()[0] 
        b = data.my_b[[0]].detach().cpu().numpy().flatten()
        
        cone_dict = {'z': data.num_zero_cone, 'l': data.num_linear_cone}
        scs_data = {'P': csc_matrix(P), 'c': c, 'A': csc_matrix(A), 'b': b, 'cone': cone_dict}
    
        # cold start
        if scs_default:
            solver = scs.SCS(scs_data, cone_dict, verbose=False, normalize=use_normalize, scale=scale, acceleration_lookback=acceleration_lookback, use_indirect=use_indirect, adaptive_scale=adaptive_scale)
        else:
            solver = scs.SCS(scs_data, cone_dict, verbose=False, normalize=False, 
                         scale=1.0, rho_x=1.0, alpha=1., 
                         acceleration_lookback=0, adaptive_scale=False)
        result = solver.solve()
        iter_orig.append(result['info']['iter'])
        time_orig.append(1e-3*(result['info']['solve_time'] + result['info']['setup_time']))
        scale_updates_orig.append(result['info']['scale_updates'])   
        obj_true.append(result['info']['pobj'])
        time_setup.append(1e-3*result['info']['setup_time'])
        acc_accept_orig.append(result['info']['accepted_accel_steps'])
        acc_rej_orig.append(result['info']['rejected_accel_steps'])
    
    # fromulate the dataframe
    df = pd.DataFrame({'instance': test_indices, 'obj_pred': obj_pred, 'obj_true': obj_true, 'primal_dist': primal_dist, 'dual_dist': dual_dist, 
                   'eq_dist_pred': eq_dist_pred, 'ineq_dist_pred': ineq_dist_pred, 'time_pred': time_pred, 
                   'iter_orig': iter_orig, 'iter_ws': iter_ws, 'time_orig': time_orig, 'time_ws': time_ws, 
                   'scale_updates_orig': scale_updates_orig, 'scale_updates_ws': scale_updates_ws,
                })

    df['time_ratio'] = (df['time_orig'] - df['time_ws'])/df['time_orig']
    df['iter_ratio'] = (df['iter_orig'] - df['iter_ws'])/df['iter_orig']
    df['time_total'] = df['time_pred'] + df['time_ws']
    df['time_ratio_total'] = (df['time_orig'] - df['time_total'])/df['time_orig']
    df.to_csv(os.path.join(args['runId'], args['probType']+ "_" + args['timeMode'] + ".csv"), index=False)
    
    
    print("Avg.Iter Orig: {:.3f}|\t Avg.Iter Ws: {:.3f}".format(df['iter_orig'].mean(), df['iter_ws'].mean()))
    print("Avg.Time Orig: {:.3f}|\t Avg.Time Ws: {:.3f}|\t Avg.Time Pred: {:.3f}|\t Avg.Time Total: {:.3f}".format(df['time_orig'].mean(), df['time_ws'].mean(), df['time_pred'].mean(), df['time_total'].mean()))
    print("---------------------------------------------------------------------")
    print("Ratio|\t solve time: {:.3f}|\t iteration: {:.3f}|\t total time: {:.3f}".format(df['time_ratio'].mean(), df['iter_ratio'].mean(), df['time_ratio_total'].mean()))
    print("OBJ Pred: {:.3f}|\t OBJ True: {:.3f}".format(df['obj_pred'].mean(), df['obj_true'].mean()))
    print("Setup Time: ", np.mean(time_setup))
    
    
    print("Acc. Accept Orig: {:.3f}|\t Acc. Rej Orig: {:.3f}".format(np.mean(acc_accept_orig), np.mean(acc_rej_orig)))
    print("Acc. Accept Ws: {:.3f}|\t Acc. Rej Ws: {:.3f}".format(np.mean(acc_accept_ws), np.mean(acc_rej_ws)))
    print("Scale Updates Orig: {:.3f}|\t Scale Updates Ws: {:.3f}".format(np.mean(scale_updates_orig), np.mean(scale_updates_ws)))
if __name__=='__main__':
    main()