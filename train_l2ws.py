import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

import numpy as np
import pickle
import gzip
import time
import os
import argparse

from utils_l2ws import SimpleProblem, NN_l2ws
import default_args

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description='SCS_unroll')
    parser.add_argument('--probType', type=str, default='emnist', 
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
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--train_unrolls', type=int)

    prefix = ""
    args = parser.parse_args()
    args = vars(args) # change to dictionary
    defaults = default_args.l2ws_default_args(args['probType'])
    for key in defaults.keys():
        if args[key] is None:
            args[key] = defaults[key]

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if 'simple_rhs' in prob_type:
        n_var = int(args['probType'].split('_')[2])
        args['simpleVar'] = n_var
        args['simpleIneq'] = n_var // 2
        args['simpleEq'] = n_var // 2
        filepath = os.path.join(prefix + 'datasets', 'simple_rhs', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
        k = args['train_unrolls']
        prob_type = prob_type + '_{}'.format(k)
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
    P, c, A, b, G, h, X, Y, S, ITER, OBJ_VAL, W, THETA = [],[],[],[],[],[],[],[],[],[],[],[],[]
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

def train_net(filepath, args):
    print(filepath)
    
    solver_step = args['lr']
    nepochs = args['epochs']
    batch_size = args['batchSize']
    num_examples = args['simpleEx']
    lambda1 = args['lambda1']
    hidden_layers = args['hiddenSize']
    frac_valid = 1/11
    num_train = int(num_examples*(1-frac_valid))
    num_valid = int(num_examples*frac_valid)
    train_indices = np.arange(num_examples)[:num_train]
    valid_indices = np.arange(num_examples)[num_train:num_train+num_valid]

    if 'rhs' in args['probType']:
        input_size = args['simpleEq']
        output_size = args['simpleVar'] + args['simpleEq'] + args['simpleIneq']

    solver_net = NN_l2ws(input_size, output_size, hidden_layers)
    solver_net.to(DEVICE)
    solver_opt = optim.AdamW(solver_net.parameters(), lr=solver_step)
    best_valid_loss = float('inf')
    supervised = args['supervised']
    train_unrolls = args['train_unrolls']
    
    if args['supervised']:
        checkpoint_path = 'checkpoint_' + args['probType'] + '_' + str(args['train_unrolls']) + '_reg' + '.pth'
    else:
        checkpoint_path = 'checkpoint_' + args['probType'] + '_' + str(args['train_unrolls']) + '_fp' + '.pth'

    early_stop_step = 0    
    train_step = 0

    
    for i in range(nepochs):
        # randomly shuffle the elements in numpy
        np.random.shuffle(train_indices)
        
        train_loss = 0
        len_train_batch = batch_size
        num_batch = max(1, len(train_indices) // batch_size)
        solver_net.train()
        for ii in range(num_batch):
            idx = train_indices[ii*len_train_batch:(ii+1)*len_train_batch]
            # print(idx)
            data = load_data(filepath, idx)
            idx = np.arange(len(idx))
            W_label = data.W.to(DEVICE)
            theta = data.theta.to(DEVICE)
            q = data.q.to(DEVICE)
            M = data.M.to(DEVICE)
            
            solver_opt.zero_grad()
            start_time = time.time()
            w = solver_net(theta)
            # take train_unrolls steps of fixed point iteration
            w, w_prev, eta, eta_prev = data.fixed_point_homo(w.unsqueeze(-1), M, q, k=train_unrolls)
            train_time = time.time() - start_time
            
            if supervised:
                loss = data.reg_loss_fn(w/eta, W_label)
            else:
                loss = data.fp_loss_fn(w/eta, w_prev/eta_prev)

            loss = loss.mean()
            loss.backward()
            
            solver_opt.step()
            train_loss += loss        
        
        X = w[:, :data.n_var] / eta
        Y = w[:, data.n_var:] / eta

        if i % 1 == 0:
            with torch.no_grad():
                print("epoch {}, train loss: {:4f}, obj: {:.4f}, eq_dist: {:4f}, ineq_dist: {:4f}, train time: {:4f}".format(i, 
                                                                                                                             train_loss.item()/num_batch, 
                                                                                                                             data.obj_fn(idx, X).mean().item(), 
                                                                                                                             data.eq_dist(idx, X).mean().item(), 
                                                                                                                             data.ineq_dist(idx, X).mean().item(), train_time))
        M.detach()
        q.detach()
        W_label.detach()
        theta.detach()
        w.detach()
        w_prev.detach()
        train_loss.detach()
        
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
                W_label = data.W.to(DEVICE)
                theta = data.theta.to(DEVICE)
                
                w = solver_net(theta)
                w, w_prev, eta, eta_prev = data.fixed_point_homo(w.unsqueeze(-1), M, q, k=train_unrolls)

                if supervised:
                    loss = data.reg_loss_fn(w/eta, W_label)
                else:
                    loss = data.fp_loss_fn(w/eta, w_prev/eta_prev)

                loss = loss.mean()
                valid_loss += loss.item()

            with open(checkpoint_path, 'wb') as f:
                    torch.save({'model_state_dict': solver_net.state_dict(),
                                'optimizer_state_dict': solver_opt.state_dict(),
                                'best_valid_loss': best_valid_loss,   
                                }, f)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                with open(checkpoint_path, 'wb') as f:
                    torch.save({'model_state_dict': solver_net.state_dict(),
                                'optimizer_state_dict': solver_opt.state_dict(),
                                'best_valid_loss': best_valid_loss,  
                                }, f)
                early_stop_step = 0
            else:
                early_stop_step += 1
                if args['earlyStop'] > 0 and early_stop_step > args['earlyStop']:
                    print("Early stopping")
                    break
                
            if i % 1 == 0:
                print("Valid: epoch {}, loss: {:.4f}, obj: {:.4f}, eq_dist: {:4f}, ws_ter: {:4f}".format(i, 
                                                                                      valid_loss / num_batch, 
                                                                                      data.obj_fn(idx, X).mean().item(), 
                                                                                      data.eq_dist(idx, X).mean().item(),
                                                                                      ws_iter/num_valid))


if __name__=='__main__':
    main()