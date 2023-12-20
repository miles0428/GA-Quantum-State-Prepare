'''
this file is used to run experiments on the GA algorithm for the gaussian distribution
author : Yu-Cheng Chung
email  : ycchung@ntnu.edu.tw
date   : 2023 13 Sep

dependencies:
    GA.py
    gene.py
    gaussian.py
    transform.py
    numpy
    os
    multiprocessing

'''
import numpy as np
from GA import GA
from gaussian import gaussian
from transform import normalize_prob_distribution
from transform import normalize_state_vector
import multiprocessing as mp
from qiskit_algorithms import optimizers

mp.set_start_method('spawn',True)
#set the parameters
num_genes = mp.cpu_count()
num_qubit = 5
length_gene = 70
mutation_rate = 0.1
cpu_count = mp.cpu_count()//2
path = 'data/gaussian/'
optimizer = optimizers.SPSA(maxiter=1500)
optimizer2 = optimizers.COBYLA(maxiter=1500)
maxiter = 100
miniter = 10
threshold = 0.6
cpu_count = mp.cpu_count()//2
GPU = False

#set the target distribution
#generate 15 mu from 0 to 15
mu = np.linspace(0,31,8)
#generate 15 sigma from 0 to 15
sigma = np.linspace(1,20,8)
#generate the target distribution
#use mu and sigma to generate 15*15 target distribution
if __name__ == '__main__':
 for i in range(2,4):
    for j in range(0,8,1):
        target_distribution=gaussian(np.arange(2**num_qubit),mu[i],sigma[j])
        target_distribution=normalize_prob_distribution(target_distribution)
        target_statevector=normalize_state_vector(np.sqrt(target_distribution))
        #set the experiment name as 'gaussian_mu_{mu}_sigma_{sigma}'
        experiment = 'gaussian_mu_{:.2f}_sigma_{:.2f}'.format(mu[i],sigma[j])
        #do the experiment
        GA(target_statevector=target_statevector,
           num_qubit=num_qubit,
           num_genes=num_genes,
           length_gene=length_gene,
           mutation_rate=mutation_rate,
           cpu_count=cpu_count,
           path=path,
           optimizer=optimizer,
           optimizer2 = optimizer2,
           maxiter=maxiter,
           miniter=miniter,
           threshold=threshold,
           experiment=experiment,
           GPU=GPU)

        
