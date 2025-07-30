import torch
torch.set_default_dtype(torch.float64)

import numpy as np
from scipy.sparse import csc_matrix

import hashlib
import numpy as np
import scs 

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
    def __init__(self, data, valid_num=100, test_num=100, calc_X=False, device="cpu"):
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


                  
        if data['l'] is not None:
            self.l = torch.tensor(data['l'])
        else:
            self.l = None
        
        if data['u'] is not None:
            self.u = torch.tensor(data['u'])
        else:
            self.u = None
        
        self.P = 0.5 * (self.P + torch.transpose(self.P, 1, 2))
        
        self.n_samples = self.P.shape[0]
        self.n_var = self.P.shape[1]
        self.n_eq = self.A.shape[1]
        self.n_ineq = self.G.shape[1]
        self.train_samples = np.arange(self.n_samples - valid_num - test_num)
        self.valid_samples = np.arange(self.n_samples - valid_num - test_num, self.n_samples - test_num)
        self.test_samples = np.arange(self.n_samples - test_num, self.n_samples)


        R = torch.ones_like(self.S)
        R[:, :self.n_eq] = 1/(1000*1e-6)
        R[:, self.n_eq:] = 1/(1e-6)
        if 'W' in data.keys():
            self.W = torch.tensor(data['W'], dtype=torch.float64)
        else:
            # W = [x, s+y]
            self.W = torch.cat([self.X, self.S/R + self.Y], dim=1)

        del data

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
        self.device = device
        self.c = self.c.unsqueeze(-1)
        self.b = self.b.unsqueeze(-1)
        self.q = self.q.unsqueeze(-1)

    def loss_fn(self, u, v):
        tmp = self.M.bmm(u) + self.q - v
        return torch.sum(tmp * tmp, dim=(1,2)) / 2
    
    def rel_primal_residual(self, X, norm=np.inf):
        device = X.device
        self.my_A = self.my_A.to(device)
        self.my_b = self.my_b.to(device)
        primal = self.my_b.unsqueeze(-1) - self.my_A.bmm(X.unsqueeze(-1))
        primal[:, self.n_eq:, :] = torch.clamp(primal[:, self.n_eq:, :], min=0)
        norm_b = torch.linalg.norm(self.my_b.unsqueeze(-1), ord=norm, dim=(1,2)).cpu().numpy()
        norm_Ax = torch.linalg.norm(self.my_A.bmm(X.unsqueeze(-1)), ord=norm, dim=(1,2)).cpu().numpy()        
        return torch.linalg.norm(primal, ord=norm, dim=(1,2)) / torch.max(torch.tensor(1.0, device=device),
                                                                          torch.max(torch.tensor(norm_b, device=device),
                                                                                    torch.tensor(norm_Ax, device=device)))
    def rel_dual_residual(self, X, Y, norm=np.inf):
        device = X.device
        self.P = self.P.to(device)
        self.c = self.c.to(device)
        dual = self.P.bmm(X.unsqueeze(-1)) + self.my_A.transpose(1, 2).bmm(Y.unsqueeze(-1)) + self.c
        norm_Px = torch.linalg.norm(self.P.bmm(X.unsqueeze(-1)), ord=norm, dim=(1,2)).cpu().numpy()
        norm_Ay = torch.linalg.norm(self.my_A.transpose(1, 2).bmm(Y.unsqueeze(-1)), ord=norm, dim=(1,2)).cpu().numpy()
        norm_c = torch.linalg.norm(self.c, ord=norm, dim=(1,2)).cpu().numpy()
        return torch.linalg.norm(dual, ord=norm, dim=(1,2)) / torch.max(torch.tensor(1.0, device=device),
                                                                        torch.max(torch.tensor(norm_Px, device=device),
                                                                                  torch.max(torch.tensor(norm_Ay, device=device),
                                                                                            torch.tensor(norm_c, device=device))))
    
    def DR_gd_torch(self, max_iter=1, tol=1e-4, w=None, u_tilde=None):
        if w is None:
            w = torch.zeros((self.M.shape[1], 1)).repeat(self.M.shape[0], 1, 1)
        if u_tilde is None:
            u_tilde = torch.zeros((self.M.shape[1], 1)).repeat(self.M.shape[0], 1, 1)
            
        for i in range(max_iter):      
            w = u_tilde + self.M.bmm(u_tilde) + self.q
            u = 2 * u_tilde - w
            u = torch.cat((u[:, :self.n_var+self.n_eq], torch.clamp(u[:, self.n_var+self.n_eq:], 0)), dim=1)  # Concatenate along the second dimension
            v = u + w - 2 * u_tilde
        return u, v
                
    def __str__(self):
        return 'SimpleProblem-{}-{}-{}-{}'.format(
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
    
    
    def scs_solve(self, batch, tol=1e-4, verbose=True, accelerate=True):
        X = []
        Y = []
        S = []
        sol_time = []
        obj_val = []
        iters = []
        for i in batch:
            P = self.P[i].detach().cpu().numpy()
            c = self.c[i].squeeze(-1).detach().cpu().numpy()
            A = self.my_A[i].detach().cpu().numpy()
            b = self.my_b[i].squeeze(-1).detach().cpu().numpy()
            cone_dict = {'z': self.num_zero_cone, 'l': self.num_linear_cone}
            
            # matrix P and A are required to be sparse matrix
            data = {'P': csc_matrix(P), 'c': c, 'A': csc_matrix(A), 'b': b, 'cone': cone_dict}
            
            if accelerate:
                solver = scs.SCS(data, cone_dict, eps_abs=tol, eps_rel=tol, verbose=verbose)
            else:
                solver = scs.SCS(data, cone_dict, eps_abs=tol, eps_rel=tol, verbose=verbose, 
                                 acceleration_lookback=0, normalize=False, adaptive_scale=False,
                                 rho_x=1.0, scale=1.0, alpha=1.0)
                
            results = solver.solve()
            if results['info']['status'] == 'solved':
                X.append(results['x'])
                Y.append(results['y'])
                S.append(results['s'])
                obj_val.append(results['info']['pobj'])
            else:
                X.append(np.ones(self.n_var) * np.nan)
                Y.append(np.ones(self.M.shape[1] - self.n_var) * np.nan)
                S.append(np.ones(self.M.shape[1] - self.n_var) * np.nan)
                obj_val.append(np.nan)
            sol_time.append((results['info']['setup_time'] + results['info']['solve_time']) * 1e-3) # the time is reported in milliseconds
            iters.append(results['info']['iter'])
        return torch.tensor(X), torch.tensor(Y), torch.tensor(S), torch.tensor(iters), torch.tensor(sol_time), torch.tensor(obj_val)

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



