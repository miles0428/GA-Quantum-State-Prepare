import drawer
import numpy as np
import GA
import gene
import os
import transform

#for each experiment in the folder
path='high-level-gate-4/'
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

save_path='data/gaussian-diff-gene-4/figure/'
os.makedirs(save_path,exist_ok=True)
for i,o in enumerate(experiment):
    file_name_distribution = f'{save_path}/distribution_mu_%.02f_sigma_%.02f.svg'%(mu[i],sigma[i])
    file_name_depth_change = f'{save_path}/depth_change_mu_%.02f_sigma_%.02f.svg'%(mu[i],sigma[i])
    file_name_circuit = f'{save_path}/circuit_mu_%.02f_sigma_%.02f.svg'%(mu[i],sigma[i])
    _gene=best_gene[i]['gene']
    theta=best_gene[i]['theta']
    num_qubit=5
    _target_distribution=target_distribution[i]
    drawer.draw_prob_distribution(_gene,theta,num_qubit,_target_distribution,
                                  file_name_distribution,method='statevector')
    #check generation number
    check_path = f'{path}/{o}'
    generation_number = 0
    for j in os.listdir(check_path):
        if j.endswith('generation'):
            generation_number+=1
    drawer.draw_depth_change_from_result(path=path,expriement=o,generation_number=generation_number,save_path=file_name_depth_change)
    circuit = best_gene[i]['circuit']
    circuit.draw(output='mpl',filename=file_name_circuit)


