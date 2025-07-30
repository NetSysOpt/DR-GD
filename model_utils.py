
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import time
torch.set_default_dtype(torch.float64)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SCS_layer(nn.Module):
    def __init__(self, emb_size, update_type="u_tilde", eta_base=None):
        super(SCS_layer, self).__init__()
        self.emb_size = emb_size
        self.update_type = update_type
        
        self.u_tilde_embedding1 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=False),
            nn.LayerNorm(self.emb_size),
        )
        self.w_embedding1 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=False),
            nn.LayerNorm(self.emb_size),
        )
        
        self.u_embedding1 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=False),
            nn.LayerNorm(self.emb_size),
        )        
        
        self.u_tilde_embedding2 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=False),
            nn.LayerNorm(self.emb_size),
        )
        self.w_embedding2 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=False),
            nn.LayerNorm(self.emb_size),                       
        )
        
        self.u_embedding2 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=False),
            nn.LayerNorm(self.emb_size),
        )       
        
        for m in self.u_tilde_embedding1:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                # m.weight.data *= 1e-2
        
        for m in self.u_tilde_embedding2:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                # m.weight.data *= 1e-3
                
        for m in self.w_embedding1:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                # m.weight.data *= 1e-2
        
        for m in self.w_embedding2:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                # m.weight.data *= 1e-3
                
        for m in self.u_embedding1:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                # m.weight.data *= 1e-3
        
        for m in self.u_embedding2:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                # m.weight.data *= 1e-3
        
        self.proj = nn.ReLU()
        
        if self.update_type == "u_tilde":
            self.eta_layer = nn.Sequential(
                nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=True),
                nn.Sigmoid()
            )
            self.eta_base = eta_base 

            for m in self.eta_layer:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    
        elif self.update_type == "w":
            self.alpha_layer = nn.Sequential(
                nn.Linear(in_features=self.emb_size, out_features=self.emb_size, bias=True),
                nn.Sigmoid()
            )
            for m in self.alpha_layer:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
        
            
    def forward(self, u_tilde, u, w, q, M, n_var, n_eq):
        
        if self.update_type == "u_tilde":    
            tmp1 = w-q
            u_tilde_next = torch.cat((self.u_tilde_embedding1(u_tilde[:, :n_var]), self.u_tilde_embedding2((u_tilde[:, n_var:]))), dim=1)
            res = u_tilde_next + torch.bmm(M, u_tilde_next) - tmp1
            grad = res + torch.bmm(M.transpose(1,2), res)
            u_tilde = u_tilde - (self.eta_layer(w) * self.eta_base) * grad
            output = u_tilde
            
        elif self.update_type == "u":
            u = 2* torch.cat((self.u_tilde_embedding1(u_tilde[:, :n_var]), self.u_tilde_embedding2(u_tilde[:, n_var:])), dim=1) - \
                (torch.cat((self.w_embedding1(w[:, :n_var]), self.w_embedding2(w[:, n_var:])), dim=1))
            output = u
        elif self.update_type == "w":
            alpha=1.5
            w = torch.cat((self.w_embedding1(w[:, :n_var]), self.w_embedding2(w[:, n_var:])), dim=1) + \
                alpha * (torch.cat((self.u_embedding1(u[:, :n_var]), self.u_embedding2(u[:, n_var:])), dim=1) - \
                torch.cat((self.u_tilde_embedding1(u_tilde[:, :n_var]), self.u_tilde_embedding2(u_tilde[:, n_var:])), dim=1))
            output = w
        
        return output
        

class SCS_unroll(nn.Module):
    def __init__(self, emb_size, num_layer, eta_base):
        super(SCS_unroll, self).__init__()
        self.num_layer = num_layer
        self.emb_size = emb_size
        self.u_tilde_layers = torch.nn.ModuleList()
        self.u_layers = torch.nn.ModuleList()
        self.w_layers = torch.nn.ModuleList()

        for _ in range(num_layer):
            self.u_tilde_layers.append(SCS_layer(emb_size, "u_tilde", eta_base=eta_base))
            self.u_layers.append(SCS_layer(emb_size, "u"))
            self.w_layers.append(SCS_layer(emb_size, "w"))
        
        self.v_layer = SCS_layer(emb_size, "v")
        
        self.output_u1 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=1, bias=False),
        )
        self.output_u2 = nn.Sequential(
            nn.Linear(in_features=self.emb_size, out_features=1, bias=False),
        )
        
        for m in self.output_u1:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                m.weight.data *= 1e-3 # simple
                # m.weight.data *= 1e-1 # qplib
                
        self.proj = nn.ReLU()
        
    def forward(self, q, M, n_var, n_eq):    
        # calculate the initial values
        u_tilde = torch.zeros((q.shape[0], q.shape[1], self.emb_size), device=DEVICE)
        #####################
        q = q.repeat_interleave(self.emb_size, dim=-1)
        w = q.clone()
        u = 2*u_tilde - w
        u = torch.cat((u[:, :n_var+n_eq], torch.clamp(u[:, n_var+n_eq:], 0)), dim=1)
        w = w + u - u_tilde
        
        for count in range(self.num_layer):
            u_tilde = self.u_tilde_layers[count](u_tilde, u, w, q, M, n_var, n_eq)
            u = self.u_layers[count](u_tilde, u, w, q, M, n_var, n_eq)
            u = torch.cat((u[:, :n_var+n_eq], torch.clamp(u[:, n_var+n_eq:], 0)), dim=1)
            # w_prev = w
            w = self.w_layers[count](u_tilde, u, w, q, M, n_var, n_eq)
        
        u = torch.cat((self.output_u1(u[:, :n_var]), self.output_u2(u[:, n_var:])), dim=1)
        u = torch.cat((u[:, :n_var+n_eq], torch.clamp(u[:, n_var+n_eq:], 0)), dim=1)
        v = None
        u_tilde = u
        return u_tilde, u, w, v

