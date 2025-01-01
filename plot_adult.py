
import pickle

dataset = 'adult'
with open('./result/'+  'Fairness.' + dataset + '.time_lst_EG.pkl', 'rb') as f:
    time_lst_EG = pickle.load(f)
with open('./result/'+  'Fairness.' + dataset + '.gnorm_lst_EG.pkl', 'rb') as f:
    gnorm_lst_EG = pickle.load(f)

with open('./result/'+  'Fairness.' + dataset + '.time_lst_EG2.pkl', 'rb') as f:
    time_lst_LEN_1 = pickle.load(f) 
with open('./result/'+  'Fairness.' + dataset + '.gnorm_lst_EG2.pkl', 'rb') as f:
    gnorm_lst_LEN_1 = pickle.load(f)

with open('./result/'+  'Fairness.' + dataset + '.time_lst_LEN.pkl', 'rb') as f:
    time_lst_LEN_10 = pickle.load(f)

with open('./result/'+  'Fairness.' + dataset + '.gnorm_lst_LEN.pkl', 'rb') as f:
    gnorm_lst_LEN_10 = pickle.load(f)

import matplotlib.pyplot as plt 

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
plt.savefig('./img/'+  'Fairness.' + dataset + '.png')
plt.savefig('./img/'+  'Fairness.' + dataset + '.pdf', format = 'pdf')

    
