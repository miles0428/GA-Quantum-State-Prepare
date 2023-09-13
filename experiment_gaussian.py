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

#set the parameters
num_genes = mp.cpu_count()
num_qubit = 4
length_gene = 25
mutation_rate = 0.1
cpu_count = mp.cpu_count()
path = 'data/gaussian/'
optimizer = optimizers.SPSA(maxiter=1000)
iter = 50
threshold = 0.90
num_types = 15

#set the target distribution
#generate 15 mu from 0 to 15
mu = np.linspace(0,15,15)
#generate 15 sigma from 0 to 15
sigma = np.linspace(0,15,15)
#generate the target distribution
#use mu and sigma to generate 15*15 target distribution
for i in range(15):
    for j in range(15):
        target_distribution=gaussian(np.arange(2**num_qubit),mu[i],sigma[j])
        target_distribution=normalize_prob_distribution(target_distribution)
        target_statevector=normalize_state_vector(np.sqrt(target_distribution))
        #set the experiment name as 'gaussian_mu_{mu}_sigma_{sigma}'
        experiment = f'gaussian_mu_{mu[i]}_sigma_{sigma[j]}'
        #do the experiment
        GA(target_statevector=target_statevector,
           num_qubit=num_qubit,
           num_genes=num_genes,
           length_gene=length_gene,
           mutation_rate=mutation_rate,
           cpu_count=cpu_count,
           path=path,
           optimizer=optimizer,
           iter=iter,
           threshold=threshold,
           num_types=num_types,
           experiment=experiment)
        









