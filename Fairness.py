# Codes adapted from

import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import time 
import torch 
from scipy.io import loadmat, savemat

parser = argparse.ArgumentParser(description='Minimax Optimization.')
parser.add_argument('--training_time',  type=float, default=10.0, help='total training time')
parser.add_argument('--dataset', type=str, default='adult', help='dataset: adult / lawschool') 
parser.add_argument('--M', type=float, default=10.0, help='parameter M for ARE')
parser.add_argument('--LazyM', type=float, default=10.0, help='parameter M for LEAN')
parser.add_argument('--seed',  type=int, default=42, help='seed of random number')
parser.add_argument('--eta', type=float, default=1e-1, help='stepsize of extra gradient')
parser.add_argument('--max_iter',  type=int, default=10, help='maximum iteration for cubic subsolver')
parser.add_argument('--eps',  type=float, default=1e-8, help='precision for cubic subsolver')

# Parameters in problem setting
parser.add_argument('--lamb',  type=float, default=1e-4)
parser.add_argument('--gamma',  type=float, default=1e-4)
parser.add_argument('--beta',  type=float, default=0.5)
args = parser.parse_args()

device=torch.device("cuda:0")

def load_adult():
    m = loadmat('./Data/a9a.mat')
    A0=np.array(m['A']).astype("float")
    b=np.array(m['b']).astype("float")
    c=A0[:,71]
    A=np.hstack((A0[:,:71],A0[:,72:]))
    c[np.where(c==0)[0]]=-1
    c=c.reshape(A0.shape[0],1)
    A=torch.from_numpy(A).double()
    b=torch.from_numpy(b).double()
    c=torch.from_numpy(c).double()
    return A,b,c

def load_lawschool():
    m = loadmat('./Data/LSTUDENT_DATA1.mat')
    A=np.array(m['A']).astype("float")
    b=np.array(m['b']).astype("float")
    b[np.where(b==0)[0]]=-1
    c=np.array(m['c']).astype('float')
    c[np.where(c==0)[0]]=-1
    A=torch.from_numpy(A).double()
    b=torch.from_numpy(b).double()
    c=torch.from_numpy(c).double()
    return A,b,c 

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

set_seed(args.seed)

class Fairness_Learning:
    def __init__(self,A,b,c,lamb,gamma,beta):
        self.m,self.dim=A.shape
        self.d=self.dim+1
        self.gamma=gamma
        self.beta=beta
        self.A=A.to(device)
        self.lamb=lamb
        self.b=b.to(device)
        self.c=c.to(device)


    def GDA_field(self,z):
        x=z[:self.dim]
        y=z[self.dim]
        Ax = self.A @ x
        p1=1/(1+torch.exp(self.b*Ax))
        p2=1/(1+torch.exp(self.c*(Ax)*y))
        gx=-self.A.T@(p1*self.b/self.m)+self.A.T@(self.c*y*p2*self.beta/self.m)+2*self.lamb*x
        gy=torch.sum((Ax)*self.c*self.beta*p2/self.m)-2*self.gamma*y
        g=torch.zeros(self.d,1).double().to(device)
        g[:self.dim]=gx
        g[self.dim]=-gy
        return g

    def Jacobian_GDA_field(self,z):
        x=z[:self.dim]
        y=z[self.dim]
        multt=self.A@x
        p1=1/(1+torch.exp(self.b*multt))
        p2=1/(1+torch.exp(self.c*(multt)*y))
        Hxx=self.A.T@(p1*(1-p1)/self.m*self.A)-self.A.T@(p2*(1-p2)*self.beta/self.m*y**2*self.A)+2*self.lamb*torch.eye(self.dim).double().to(device)
        Hyy=-torch.sum(p2*(1-p2)/self.m*self.beta*(self.A@x)**2)-2*self.gamma
        Hxy=-self.A.T@((self.A@x)*y*p2*(1-p2)*self.beta/self.m)+self.A.T@(p2*self.beta/self.m*self.c)
        H=torch.eye(self.d).double().to(device)
        H[:self.dim,:self.dim]=Hxx
        H[self.dim,self.dim]=-Hyy
        H[:self.dim,self.dim]=Hxy.reshape(-1)
        H[self.dim,:self.dim]=-Hxy.reshape(-1)
        return H

if args.dataset == 'adult':
    A,b,c = load_adult()
elif args.dataset == 'lawschool':
    A,b,c = load_lawschool()
else:
    raise ValueError('should use the correct dataset')

oracle = Fairness_Learning(A,b,c,lamb=args.lamb,gamma=args.gamma,beta=args.beta)
z0=torch.zeros(oracle.d,1).double().to(device)

# Extra Gradient

def EG(oracle, z0, args):
    time_lst_EG = []
    gnorm_lst_EG = []
    ellapse_time = 0.0
    z = z0
    i = 0
    while True:
        start = time.time()
        gz =  oracle.GDA_field(z)
        z_half = z - args.eta * gz
        gz_half = oracle.GDA_field(z_half)
        z = z - args.eta * gz_half
        end = time.time()
        ellapse_time += end - start
        
        if ellapse_time > args.training_time:
            break 
        time_lst_EG.append(ellapse_time)

        gnorm = torch.linalg.norm(gz_half).item()
        gnorm_lst_EG.append(gnorm)

        if i % 10 == 0:
            print('EG: Epoch %d |  gradient norm %.4f' % (i, gnorm))

        i = i + 1
    return time_lst_EG, gnorm_lst_EG

# LEAN
def LEAN(oracle, z0, args, M, m):
    time_lst_LEAN = []
    gnorm_lst_LEAN = []
    ellapse_time = 0.0
    z = z0
    z_half = z0
    i = 0
    while True:
        start = time.time()
        gz =  oracle.GDA_field(z).cdouble()

        # Compute the spectral decomposition of the snapshot Hessian
        if i % m == 0:
            Hz = oracle.Jacobian_GDA_field(z)
            Lambda, Q = torch.linalg.eig(Hz)
            Lambda = torch.reshape(Lambda, (oracle.d, 1))
            invQ = torch.linalg.inv(Q)

        # Apply univariate Newton method to solve the cubic sub-problem
        r = torch.linalg.norm(z_half - z)
        tilde_gz = invQ @ gz 
        
        for j in range(args.max_iter):
            inverse_reg_Gamma = torch.pow(Lambda + M * r, -1)
            s = (Q @ (inverse_reg_Gamma * tilde_gz)).real
            s_norm = torch.linalg.norm(s)
            varphi = r - s_norm    
            varphi_prime = 1 + (M / s_norm * tilde_gz.conj().T @ ( torch.pow(inverse_reg_Gamma, 3) * tilde_gz)).real

            r = r - varphi / varphi_prime
            if torch.abs(varphi) < args.eps:
                print('Epoch %d | Univariate Newton succeed in iteration %d' % (i, j))
                break
        
        inverse_reg_Gamma = torch.pow(Lambda + M * r, -1)
        z_half = (z - Q @ ( inverse_reg_Gamma * tilde_gz)).real

        gamma = M * torch.linalg.norm(z_half - z)
        eta = 1 / gamma 

        gz_half = oracle.GDA_field(z_half)
        z = z - eta * gz_half

        end = time.time()
        ellapse_time += end - start
        if ellapse_time > args.training_time:
            break
        time_lst_LEAN.append(ellapse_time)

        gnorm = torch.linalg.norm(gz_half).item()
        gnorm_lst_LEAN.append(gnorm)

        if i % 10 == 0:
            print('LEAN (m=%d): Epoch %d  | gradient norm %.4f' % (m, i, gnorm))
        i = i + 1
    return time_lst_LEAN, gnorm_lst_LEAN

time_lst_ARE, gnorm_lst_ARE = LEAN(oracle, z0, args, M=args.M, m=1)
time_lst_EG, gnorm_lst_EG = EG(oracle, z0, args)

d = oracle.d
time_lst_LEAN, gnorm_lst_LEAN = LEAN(oracle, z0, args, M=args.LazyM, m=d)

# Plot the results time vs. gnorm 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font', size=21)
plt.figure()
plt.grid()
plt.yscale('log')
plt.plot(time_lst_EG, gnorm_lst_EG, '-.b', label='EG', linewidth=3)
plt.plot(time_lst_ARE, gnorm_lst_ARE, ':r', label='ARE', linewidth=3)
plt.plot(time_lst_LEAN, gnorm_lst_LEAN, '-k', label='LEAN', linewidth=3)
plt.legend(fontsize=23, loc='lower right')
plt.tick_params('x',labelsize=21)
plt.tick_params('y',labelsize=21)
plt.ylabel('grad. norm')
plt.xlabel('time')
plt.tight_layout()
plt.savefig('./img/'+  'Fairness.' + args.dataset + '.' + str(args.beta) + '.png')
plt.savefig('./img/'+  'Fairness.' + args.dataset + '.' + str(args.beta) + '.pdf', format = 'pdf')
