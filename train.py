# try:
#     import waitGPU
#     waitGPU.wait(utilization=50, memory_ratio=0.5, available_memory=5000, interval=9, nproc=10000, ngpu=1)
# except ImportError:
#     pass


import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

import scs
from scipy.sparse import csc_matrix
import numpy as np
import pickle
import gzip
import time
import os
import argparse

from utils import SimpleProblem
import default_args
from model_utils import SCS_unroll

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
valid = True

def main():
    parser = argparse.ArgumentParser(description='SCS_unroll')
    parser.add_argument('--probType', type=str, default='qplib_4270_0.1', #default='qplib_8616',
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
        help='base learning rate for the optimizer')

    prefix = ""
    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.method_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]


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
    
    print(args)

    print(args['probType'])
    print('file path: ', filepath)
    train_net(filepath, args)


def load_data(filepath, index):
    P, c, A, b, G, h, X, Y, S, ITER, OBJ_VAL = [],[],[],[],[],[],[],[],[],[],[]
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

def train_net(filepath, args):
    print(filepath)
    
    solver_step = args['lr']
    nepochs = args['epochs']
    batch_size = args['batchSize']
    num_examples = args['simpleEx']
    lambda1 = args['lambda1']
    frac_valid = 1/11
    num_train = int(num_examples*(1-frac_valid))
    num_valid = int(num_examples*frac_valid)
    train_indices = np.arange(num_examples)[:num_train]
    valid_indices = np.arange(num_examples)[num_train:num_train+num_valid]

    solver_net = SCS_unroll(args['embSize'], args['numLayers'], args['etaBase'])
    solver_net.to(DEVICE)
    solver_opt = optim.AdamW(solver_net.parameters(), lr=solver_step)
    best_valid_loss = float('inf')
    best_valid_iter = float('inf')
    

    
    if args['probType'] == 'simple':
        checkpoint_path = 'checkpoint_' + args['probType'] + '_' + str(args['simpleVar']) +'.pth'
    else:
        checkpoint_path = 'checkpoint_' + args['probType'] + '.pth'
    
    early_stop_step = 0    
    checkpoint_path = 'checkpoint_' + args['probType'] + '.pth'

    
    for i in range(nepochs):
        # randomly shuffle the elements in numpy
        np.random.shuffle(train_indices)
        
        train_loss = 0
        len_train_batch = batch_size
        num_batch = max(1, len(train_indices) // batch_size)
        solver_net.train()
        for ii in range(num_batch):
            idx = train_indices[ii*len_train_batch:(ii+1)*len_train_batch]
            data = load_data(filepath, idx)
            idx = np.arange(len(idx))
            X_label = data.X.to(DEVICE)
            Y_label = data.Y.to(DEVICE)
            q = data.q.to(DEVICE)
            M = data.M.to(DEVICE)
            
            solver_opt.zero_grad()
            start_time = time.time()
            _, u, _, _ = solver_net(q, M, data.n_var, data.n_eq) 
            train_time = time.time() - start_time
            
            X = u[:, :data.n_var].squeeze(-1)
            Y = u[:, data.n_var:].squeeze(-1)
            
            loss = nn.MSELoss()(X.squeeze(-1), X_label)*lambda1 + nn.MSELoss()(Y.squeeze(-1), Y_label)
            loss.backward()
            
            solver_opt.step()
            train_loss += loss
            

        if i % 1 == 0:
            X = X.detach()
            Y = Y.detach()
            torch.cuda.empty_cache()
            with torch.no_grad():
                print("epoch {}, train loss: {:4f}, obj: {:.4f}, eq_dist: {:4f}, ineq_dist: {:4f}, train time: {:4f}".format(i, 
                                                                                                                             train_loss.item()/num_batch, 
                                                                                                                             data.obj_fn(idx, X).mean().item(), 
                                                                                                                             data.eq_dist(idx, X).mean().item(), 
                                                                                                                             data.ineq_dist(idx, X).mean().item(), train_time))
                

        loss.detach()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            solver_net.eval()
            num_batch = max(1, num_valid // batch_size) 
            valid_loss = 0
            ws_iter = 0
            for ii in range(num_batch):
                idx = valid_indices[ii*batch_size:(ii+1)*batch_size]
                data = load_data(filepath, idx)
                idx = np.arange(len(idx))
                M = data.M.to(DEVICE)
                q = data.q.to(DEVICE)
                X_label = data.X.to(DEVICE)
                Y_label = data.Y.to(DEVICE)

                _, u, _, v = solver_net(q, M, data.n_var, data.n_eq) 
                X = u[:, :data.n_var].squeeze(-1)
                Y = u[:, data.n_var:].squeeze(-1)
                u = u.detach()
                q = q.detach()
                M = M.detach()
                X = X.detach()
                Y = Y.detach()
                torch.cuda.empty_cache()
                
                loss = nn.MSELoss()(X.squeeze(-1), X_label)*lambda1 + nn.MSELoss()(Y.squeeze(-1), Y_label) 
                valid_loss += loss.item()
                
                if valid:
                    for iii in range(X.shape[0]):
                        P = data.P[[iii]].detach().cpu().numpy()[0] 
                        c = data.c[[iii]].detach().cpu().numpy().flatten()
                        A = data.my_A[[iii]].detach().cpu().numpy()[0] 
                        b = data.my_b[[iii]].detach().cpu().numpy().flatten()

                        x_ws = X[iii].detach().cpu().numpy().flatten()
                        y_ws = Y[iii].detach().cpu().numpy().flatten()
                        s_ws = b - A @ x_ws

                        cone_dict = {'z': data.num_zero_cone, 'l': data.num_linear_cone}
                        scs_data = {'P': csc_matrix(P), 'c': c, 'A': csc_matrix(A), 'b': b, 'cone': cone_dict}

                        # warm start
                        solver_ws = scs.SCS(scs_data, cone_dict, eps_abs=1e-4, eps_rel=1e-4, verbose=False, 
                                    acceleration_lookback=0, normalize=False, adaptive_scale=False,
                                    rho_x=1.0, scale=1.0, alpha=1.)
                        result_ws = solver_ws.solve(warm_start=True, x=x_ws, y=y_ws, s=s_ws)
                        ws_iter += result_ws['info']['iter']
                else:
                    # valid_loss = float('inf')
                    ws_iter = float('inf')
                    # with open(checkpoint_path, 'wb') as f:
                    #     torch.save({'model_state_dict': solver_net.state_dict(),
                    #             'optimizer_state_dict': solver_opt.state_dict(),
                    #             'best_valid_loss': best_valid_loss,    # save loss if needed
                    #             }, f)

                
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                print("Saving Checkpoint")
                with open(checkpoint_path, 'wb') as f:
                    torch.save({'model_state_dict': solver_net.state_dict(),
                                'optimizer_state_dict': solver_opt.state_dict(),
                                'best_valid_loss': best_valid_loss,    # save loss if needed
                                }, f)
                early_stop_step = 0


            if i % 1 == 0:
                print("Valid: epoch {}, loss: {:.4f}, obj: {:.4f}, eq_dist: {:4f}, ws_ter: {:4f}".format(i, 
                                                                                      valid_loss / num_batch, 
                                                                                      data.obj_fn(idx, X).mean().item(), 
                                                                                      data.eq_dist(idx, X).mean().item(),
                                                                                      ws_iter/num_valid))
                        
                
        if ws_iter/num_valid < best_valid_iter:
            print("Saving Checkpoint")
            best_valid_iter = ws_iter/num_valid
            checkpoint_path_bak = '' + '/checkpoint_' + args['probType'] + '.pth'
            with open(checkpoint_path_bak, 'wb') as f:
                    torch.save({'model_state_dict': solver_net.state_dict(),
                                'optimizer_state_dict': solver_opt.state_dict(),
                                'best_valid_loss': best_valid_loss,    # save loss if needed
                                }, f)
        
        if i % 10 == 0:
            checkpoint_path_bak = '' + '/checkpoints_' + args['probType'] + '/checkpoint_' + args['probType'] + '_' + str(i) + '.pth'
            with open(checkpoint_path_bak, 'wb') as f:
                    torch.save({'model_state_dict': solver_net.state_dict(),
                                'optimizer_state_dict': solver_opt.state_dict(),
                                'best_valid_loss': best_valid_loss,    # save loss if needed
                                }, f)

if __name__=='__main__':
    main()