import torch
torch.set_default_dtype(torch.float64)

import numpy as np
from scipy.sparse import csc_matrix

import hashlib
import numpy as np
import scs 

import torch
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{value} is not a valid boolean value')

def my_hash(string):
    return hashlib.sha1(bytes(string, 'utf-8')).hexdigest()

class SimpleProblem:
    """ 
        minimize_y 1/2 * x^T P x + c^Tx
        s.t.       Ax =  b
                   Gx <= h
                   l <= x <= u
    """
    def __init__(self, data, valid_num=100, test_num=100):
        self.P = torch.tensor(data['P'], dtype=torch.float64) 
        self.c = torch.tensor(data['c'], dtype=torch.float64) 
        self.A = torch.tensor(data['A'], dtype=torch.float64) 
        self.b = torch.tensor(data['b'], dtype=torch.float64) 
        self.G = torch.tensor(data['G'], dtype=torch.float64) 
        self.h = torch.tensor(data['h'], dtype=torch.float64) 
        
        self.X = torch.tensor(data['X'], dtype=torch.float64)
        self.Y = torch.tensor(data['Y'], dtype=torch.float64)
        self.S = torch.tensor(data['S'], dtype=torch.float64)
        self.OBJ_VAL = torch.tensor(data['obj'], dtype=torch.float64)
        self.ITER = torch.tensor(data['iter'], dtype=torch.float64)
        self.theta = torch.tensor(data['theta'], dtype=torch.float64)
        self.W = torch.tensor(data['W'], dtype=torch.float64)
        
        if data['l'] is not None:
            self.l = torch.tensor(data['l'])
        else:
            self.l = None
        
        if data['u'] is not None:
            self.u = torch.tensor(data['u'])
        else:
            self.u = None
        del data

        self.P = 0.5 * (self.P + torch.transpose(self.P, 1, 2))
        
        self.n_samples = self.P.shape[0]
        self.n_var = self.P.shape[1]
        self.n_eq = self.A.shape[1]
        self.n_ineq = self.G.shape[1]
        self.train_samples = np.arange(self.n_samples - valid_num - test_num)
        self.valid_samples = np.arange(self.n_samples - valid_num - test_num, self.n_samples - test_num)
        self.test_samples = np.arange(self.n_samples - test_num, self.n_samples)

        if self.A.shape[1] == 0 and self.b.shape[1] == 0:
            self.my_A = torch.zeros((self.n_samples, 1, self.n_var))
            self.my_b = torch.zeros((self.n_samples, 1, ))
        else:
            self.my_A = self.A
            self.my_b = self.b
        
        if self.G.shape[1] and self.h.shape[1]:
            self.my_A = torch.cat([self.my_A, self.G], axis=1)
            self.my_b = torch.cat([self.my_b, self.h], axis=1)
    
        if self.l is not None:
            # for each sample, extract the indeces with finite values
            lb_idx = [np.arange(self.n_var)[self.l[i].cpu().numpy() != -np.inf] for i in range(self.n_samples)]
            A_lb = torch.tensor(np.array([-np.eye(self.n_var)[lb_idx[i], :] for i in range(self.n_samples)]))
            b_lb = torch.tensor(np.array([-self.l[i][lb_idx[i]].cpu().numpy() for i in range(self.n_samples)]))
            self.my_A = torch.cat([self.my_A, A_lb], axis=1)
            self.my_b = torch.cat([self.my_b, b_lb], axis=1)
    
        if self.u is not None:
            ub_idx = [np.arange(self.n_var)[self.u[i].cpu() != np.inf] for i in range(self.n_samples)]
            A_ub = torch.tensor(np.array([np.eye(self.n_var)[ub_idx[i], :] for i in range(self.n_samples)]))
            b_ub = torch.tensor(np.array([self.u[i][ub_idx[i]].cpu().numpy() for i in range(self.n_samples)]))
            self.my_A = torch.cat([self.my_A, A_ub], axis=1)
            self.my_b = torch.cat([self.my_b, b_ub], axis=1)
        
        if self.A.shape[1] == 0 and self.b.shape[1] == 0:
            self.my_A = self.my_A[:, 1:, :]
            self.my_b = self.my_b[:, 1:]
        
        self.num_zero_cone = self.n_eq
        self.num_linear_cone = self.my_A.shape[1] - self.num_zero_cone
        
        # defining the matrix M and q
        self.M = torch.zeros((self.n_samples, self.n_var + self.my_A.shape[1], self.n_var + self.my_A.shape[1]))
        self.q = torch.zeros((self.n_samples, self.n_var + self.my_A.shape[1]))
        self.M[:, :self.n_var, :self.n_var] = self.P
        self.M[:, :self.n_var, self.n_var:] = torch.transpose(self.my_A, 1, 2)
        self.M[:, self.n_var:, :self.n_var] = -self.my_A
        self.q[:, :self.n_var] = self.c
        self.q[:, self.n_var:] = self.my_b
        

        ### For Pytorch
        self.c = self.c.unsqueeze(-1)
        self.b = self.b.unsqueeze(-1)
        self.q = self.q.unsqueeze(-1)
        
        

                
    def __str__(self):
        return 'SimpleProblemVec-{}-{}-{}-{}'.format(
            str(self.n_var), str(self.n_ineq), str(self.n_eq), str(self.n_samples)
        )
    
    def obj_fn(self, batch, X):
        device = X.device
        self.P = self.P.to(device)
        self.c = self.c.to(device)
        if len(X.shape) == 2:
            obj = 0.5 * torch.sum(X.unsqueeze(-1) * (self.P[batch] @ X.unsqueeze(-1)), dim=(1, 2)) + torch.sum(self.c[batch] * X.unsqueeze(-1), dim=(1, 2))
        else:
            obj =  0.5 * torch.sum(X * (self.P[batch] @ X), dim=(1,2)) + torch.sum(self.c[batch] * X, dim=(1,2))
        
        # move the cpu to save the memory
        self.P = self.P.cpu()
        self.c = self.c.cpu()
        return obj
        
    def eq_resid(self, batch, X):
        device = X.device
        self.A = self.A.to(device)
        self.b = self.b.to(device)
        if len(X.shape) == 2:
            resid = self.A[batch].bmm(X.unsqueeze(-1)) - self.b[batch]
        else:
            resid = self.A[batch].bmm(X) - self.b[batch]
        self.A = self.A.cpu()
        self.b = self.b.cpu()
        return resid
        
    def eq_dist(self, batch, X):
        if self.A.shape[1] == 0:
            return torch.zeros_like(X)
        eq_resid = self.eq_resid(batch, X)
        return torch.abs(eq_resid)
    
    def ineq_resid(self, batch, X):
        device = X.device
        self.G = self.G.to(device)
        self.h = self.h.to(device)
        if len(X.shape) == 2:
            resid = self.G[batch].bmm(X.unsqueeze(-1)) - self.h[batch].unsqueeze(-1)
        else:
            resid = self.G[batch].bmm(X) - self.h[batch].unsqueeze(-1)
        self.G = self.G.cpu()
        self.h = self.h.cpu()
        return resid
    
    def ineq_dist(self, batch, X):
        if self.G.shape[1] == 0:
            return torch.zeros_like(X)
        ineq_resid = self.ineq_resid(batch, X)
        return torch.clamp(ineq_resid, 0)
    
    def bound_dist(self, batch, X):
        if self.l is None or self.u is None:
            return torch.zeros_like(X)
        return torch.max(torch.clamp(self.l[batch] - X ,min=0), torch.clamp(X - self.u[batch], min=0))
    
    
    def fixed_point(self, w, M, q, k=1):
        device = w.device
        IM = torch.eye(M.shape[1], device=device) + M
        IM_inv = torch.linalg.inv(IM)

        for _ in range(k+1):
            u_tilde = IM_inv.bmm(w-q)
            u = 2*u_tilde - w
            u = torch.cat([u[:, :self.n_var+self.n_eq], torch.clamp(u[:, self.n_var+self.n_eq:], min=0)], dim=1)
            v = u + w - 2 * u_tilde
            w_prev = w.clone()
            w = w + 1.5*(u - u_tilde)
        
        return w.squeeze(-1), w_prev.squeeze(-1), u.squeeze(-1)
    
    def root(self, mu, eta, p, r):
        # Calculating in a batched manner for PyTorch
        a = 1 + torch.sum(r * r, dim=1)
        b = torch.sum(r * mu, dim=1) - 2 * torch.sum(r * p, dim=1) - eta
        c = torch.sum(p * (p - mu), dim=1)
    
        return (-b + torch.sqrt(b**2 - 4 * a * c)) / (2 * a)
        
    def fixed_point_homo(self, mu, M, q, k=1):
        device = M.device  # Keep the device consistent
        batch_size = M.shape[0]  # Assume M is batched

        # Initialize `mu` and `eta` for batch processing
        if mu is None:
            mu = torch.zeros(batch_size, M.shape[1], device=device)
        eta = torch.ones((batch_size,1), device=device)

        mu_prev = mu.clone()
        eta_prev = eta.clone()

        # `r` is solved per batch for compatibility
        I = torch.eye(M.shape[1], device=device).unsqueeze(0).expand_as(M)
        r = torch.linalg.solve(I + M, q)

        for i in range(k+1):
            # Calculate `u_tilde`
            p = torch.linalg.solve(I + M, mu)
            tau_tilde = self.root(mu, eta, p, r)
            z_tilde = p - r * tau_tilde.unsqueeze(-1)

            # Calculate `u`
            z = 2 * z_tilde - mu
            z = torch.cat([z[:, :self.n_var+self.n_eq], torch.clamp(z[:, self.n_var+self.n_eq:], 0)], dim=1)
            tau = torch.maximum(2 * tau_tilde - eta, torch.zeros_like(eta))

            mu_prev = mu.clone()
            eta_prev = eta.clone()
            
            # Update `w`
            mu = mu + 1.*(z - z_tilde)
            eta = eta + 1.*(tau - tau_tilde)
            
        return mu.squeeze(-1), mu_prev.squeeze(-1), eta, eta_prev
    
    def fp_loss_fn(self, w, w_prev):
        return torch.linalg.norm(w-w_prev, dim=1)

    def reg_loss_fn(self, w, w_label):
        return torch.linalg.norm(w-w_label, dim=1)
    


class NN_l2ws(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(NN_l2ws, self).__init__()
        # define the layers
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.ModuleList()
        if len(hidden_layers) == 0:
            self.layers.append(nn.Sequential(nn.Linear(input_size, output_size)))
        else:
            self.layers.append(nn.Sequential(nn.Linear(input_size, hidden_layers[0]), nn.ReLU()))
            for i in range(1, len(hidden_layers)):
                self.layers.append(nn.Sequential(nn.Linear(hidden_layers[i-1], hidden_layers[i]), nn.ReLU()))
            self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        
        # initialize the weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        