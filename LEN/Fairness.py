# Codes adapted from

import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import time 
import torch 
from scipy.io import loadmat, savemat
import scipy 
from scipy.optimize import fsolve
from libsvm.svmutil import svm_read_problem
import pickle

parser = argparse.ArgumentParser(description='Minimax Optimization.')
parser.add_argument('--training_time',  type=float, default=10.0, help='total training time')
parser.add_argument('--dataset', type=str, default='adult', help='dataset: adult / lawschool / covtype') 
parser.add_argument('--rho', type=float, default=10.0, help='parameter rho for LEN')
parser.add_argument('--seed',  type=int, default=42, help='seed of random number')
parser.add_argument('--m', type=int, default=10, help='frequency of Lazy Hessian')
parser.add_argument('--eta', type=float, default=1e-1, help='stepsize of extra gradient')
parser.add_argument('--max_iter',  type=int, default=10, help='maximum iteration for cubic subsolver')
parser.add_argument('--eps',  type=float, default=1e-8, help='precision for cubic subsolver')

# Parameters in problem setting
parser.add_argument('--lamb',  type=float, default=1e-4)
parser.add_argument('--gamma',  type=float, default=1e-4)
parser.add_argument('--beta',  type=float, default=0.5)
args = parser.parse_args()

device=torch.device("cuda:0")

def load_heart():
    idx = 1
    b, A = svm_read_problem('Data/heart_scale', return_scipy=True)

    A0 = torch.sparse_csr_tensor(
        torch.from_numpy(A.indptr),
        torch.from_numpy(A.indices),
        torch.from_numpy(A.data),
        size=A.shape
    ).to_dense().double()
    # A0=torch.from_numpy(A0).double()
    
    b=b.reshape(-1,1)
    A0 = A0.numpy()

    c=np.array(A0[:,idx])
    A=np.hstack((A0[:,:idx],A0[:,idx+1:]))
    c[np.where(c==0)[0]]=-1
    c=c.reshape(A0.shape[0],1)

    return A,b,c

def load_adult():
    m = loadmat('./Data/a9a.mat')
    A0=np.array(m['A']).astype("float")
    b=np.array(m['b']).astype("float")
    c=A0[:,71]
    A=np.hstack((A0[:,:71],A0[:,72:]))
    c[np.where(c==0)[0]]=-1
    c=c.reshape(A0.shape[0],1)
    # A=torch.from_numpy(A).double()
    # b=torch.from_numpy(b).double()
    # c=torch.from_numpy(c).double()
    return A,b,c

def load_lawschool():
    m = loadmat('./Data/LSTUDENT_DATA1.mat')
    A=np.array(m['A']).astype("float")
    b=np.array(m['b']).astype("float")
    b[np.where(b==0)[0]]=-1
    c=np.array(m['c']).astype('float')
    c[np.where(c==0)[0]]=-1
    # A=torch.from_numpy(A).double()
    # b=torch.from_numpy(b).double()
    # c=torch.from_numpy(c).double()
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
        self.lamb=lamb
        self.A = A
        self.b = b
        self.c = c 

        #self.A=A.to(device)
        #self.b=b.to(device)
        #self.c=c.to(device)


    def GDA_field(self,z):
        x=z[:self.dim]
        y=z[self.dim]
        Ax = self.A @ x
        p1=1/(1+np.exp(self.b*Ax))
        p2=1/(1+np.exp(self.c*(Ax)*y))
        gx=-self.A.T@(p1*self.b/self.m)+self.A.T@(self.c*y*p2*self.beta/self.m)+2*self.lamb*x
        gy=np.sum((Ax)*self.c*self.beta*p2/self.m)-2*self.gamma*y
        g=np.zeros((self.d,1)) #.double().to(device)
        g[:self.dim]=gx
        g[self.dim]=-gy
        return g

    def Jacobian_GDA_field(self,z):
        x=z[:self.dim]
        y=z[self.dim]
        multt=self.A@x
        p1=1/(1+np.exp(self.b*multt))
        p2=1/(1+np.exp(self.c*(multt)*y))
        Hxx=self.A.T@(p1*(1-p1)/self.m*self.A)-self.A.T@(p2*(1-p2)*self.beta/self.m*y**2*self.A)+2*self.lamb*np.eye(self.dim) #.double().to(device)
        Hyy=-np.sum(p2*(1-p2)/self.m*self.beta*(self.A@x)**2)-2*self.gamma
        Hxy=-self.A.T@((self.A@x)*y*p2*(1-p2)*self.beta/self.m)+self.A.T@(p2*self.beta/self.m*self.c)
        H=np.eye(self.d) #.double().to(device)
        H[:self.dim,:self.dim]=Hxx
        H[self.dim,self.dim]=-Hyy
        H[:self.dim,self.dim]=Hxy.reshape(-1)
        H[self.dim,:self.dim]=-Hxy.reshape(-1)
        return H

if args.dataset == 'adult':
    A,b,c = load_adult()
elif args.dataset == 'lawschool':
    A,b,c = load_lawschool()
elif args.dataset == 'heart':
    A,b,c = load_heart()
else:
    raise ValueError('should use the correct dataset')

oracle = Fairness_Learning(A,b,c,lamb=args.lamb,gamma=args.gamma,beta=args.beta)
z0=np.zeros((oracle.d,1)) #.double().to(device)

rho = args.rho 

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

        gnorm = np.linalg.norm(gz_half).item()
        gnorm_lst_EG.append(gnorm)

        if i % 10 == 0:
            print('EG: Epoch %d |  gradient norm %.4f' % (i, gnorm))

        i = i + 1
    return time_lst_EG, gnorm_lst_EG

def run_LEN(oracle, z0, args, m):
    time_lst_LEN = []
    dist_lst_LEN = []
    gnorm_lst_LEN = []
    ellapse_time = 0.0

    z = z0
    z_half = z0
    i = 0
    gamma = 1 / (m*rho)

    while True:
        start = time.time()
        gz =  oracle.GDA_field(z)

        # Compute the Schur decomposition of the snapshot Hessian
        if i % m == 0:
            Hz = oracle.Jacobian_GDA_field(z)
            U, Q = scipy.linalg.schur(Hz, output='complex')

        def func(gamma):
            return gamma - 3 / (16 * m * rho * np.linalg.norm( Q @ scipy.linalg.solve_triangular(1/gamma * np.eye(oracle.d) + U, Q.conj().T @ gz )))

        gamma = fsolve(func, gamma)

        z_half = z -  (Q @ scipy.linalg.solve_triangular(1/gamma * np.eye(oracle.d) + U, Q.conj().T @ gz )).real
        gz_half = oracle.GDA_field(z_half)
        z = z - gamma * gz_half

        end = time.time()
        ellapse_time += end - start
        if ellapse_time > args.training_time:
            break
        time_lst_LEN.append(ellapse_time)

        gnorm = np.linalg.norm(gz_half)
        gnorm_lst_LEN.append(gnorm)

        if i % 10 == 0:
            print('LEN(m=%d): Epoch %d | gradient norm %.4f' % (m, i, gnorm))
        i = i + 1
    
    return time_lst_LEN, gnorm_lst_LEN 

time_lst_EG, gnorm_lst_EG = EG(oracle, z0, args)
time_lst_LEN_1, gnorm_lst_LEN_1 = run_LEN(oracle, z0, args, m=1)
time_lst_LEN_10, gnorm_lst_LEN_10 = run_LEN(oracle, z0, args, m=args.m)

with open('./result/'+  'Fairness.' + args.dataset + '.time_lst_EG.pkl', 'wb') as f:
    pickle.dump(time_lst_EG, f)
with open('./result/'+  'Fairness.' + args.dataset + '.gnorm_lst_EG.pkl', 'wb') as f:
    pickle.dump(gnorm_lst_EG, f)

with open('./result/'+  'Fairness.' + args.dataset + '.time_lst_EG2.pkl', 'wb') as f:
    pickle.dump(time_lst_LEN_1, f)
with open('./result/'+  'Fairness.' + args.dataset + '.gnorm_lst_EG2.pkl', 'wb') as f:
    pickle.dump(gnorm_lst_LEN_1, f)

with open('./result/'+  'Fairness.' + args.dataset + '.time_lst_LEN.pkl', 'wb') as f:
    pickle.dump(time_lst_LEN_10, f)
with open('./result/'+  'Fairness.' + args.dataset + '.gnorm_lst_LEN.pkl', 'wb') as f:
    pickle.dump(gnorm_lst_LEN_10, f)

# Plot the results time vs. gnorm 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font', size=21)
plt.figure()
plt.grid()
plt.yscale('log')
plt.plot(time_lst_EG, gnorm_lst_EG, '-.b', label='EG', linewidth=3)
plt.plot(time_lst_LEN_1, gnorm_lst_LEN_1, ':r', label='EG-2', linewidth=3)
plt.plot(time_lst_LEN_10, gnorm_lst_LEN_10, '-k', label='LEN', linewidth=3)
plt.legend(fontsize=23, loc='lower right')
plt.tick_params('x',labelsize=21)
plt.tick_params('y',labelsize=21)
plt.ylabel('grad. norm')
plt.xlabel('time (s)')
plt.tight_layout()
plt.savefig('./img/'+  'Fairness.' + args.dataset + '.' + str(args.beta) + '.png')
plt.savefig('./img/'+  'Fairness.' + args.dataset + '.' + str(args.beta) + '.pdf', format = 'pdf')