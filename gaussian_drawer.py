import drawer
import numpy as np
import GA
import gene
import os
import transform

#for each experiment in the folder
path='data/gaussian/'
#find the experiment name
experiment=[]
for i in os.listdir(path):
    if i=='figure':
        continue
    if os.path.isdir(path+i):
        experiment.append(i)

mu=[]
sigma=[]

for i in experiment:
    #experiment name is like 'gaussian_mu_{mu}_sigma_{sigma}'
    mu.append(float(i.split('_')[2]))
    sigma.append(float(i.split('_')[4]))

mu=np.array(mu)
sigma=np.array(sigma)


#find the best gene
best_gene=[]
target_distribution=[]
for i in experiment:
    #experiment name is like 'gaussian_mu_{mu}_sigma_{sigma}'
    best_file_name=f'{path}/{i}/best_gene.npy'
    #find the best gene
    
    best_gene.append(np.load(best_file_name,allow_pickle=True).item())
    #find the target distribution
    target_distribution.append(transform.statevector2prob(np.load(f'{path}/{i}/target_statevector.npy',allow_pickle=True)))

save_path='data/gaussian/figure/'
os.makedirs(save_path,exist_ok=True)
for i,o in enumerate(experiment):
    file_name_distribution = f'{save_path}/distribution_mu_%.02f_sigma_%.02f.svg'%(mu[i],sigma[i])
    file_name_circuit = f'{save_path}/circuit_mu_%.02f_sigma_%.02f.svg'%(mu[i],sigma[i])
    _gene=best_gene[i]['gene']
    theta=best_gene[i]['theta']
    num_qubit=4
    _target_distribution=target_distribution[i]

    drawer.draw_prob_distribution(_gene,theta,num_qubit,_target_distribution,
                                  file_name_distribution,method='statevector')
    circuit = best_gene[i]['circuit']
    circuit.draw(output='mpl',filename=file_name_circuit)


