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
path = 'data/w-state/'
optimizer = optimizers.SPSA(maxiter=1500)
optimizer2 = optimizers.COBYLA(maxiter=1500)
maxiter = 100
miniter = 10
threshold = 0.6
cpu_count = mp.cpu_count()//2
GPU = False

def w_state(n):
    """
    Generate the w state with n qubits.
    ------------------------
    args:
        n: the number of qubits
    return:
        the w state
    ------------------------
    example:
        w_state(3) = 1/sqrt(3)(|100>+|010>+|001>)
    """
    state = np.zeros(2**n)
    for i in range(n):
        state[2**i] = 1
    return normalize_state_vector(state)


if __name__ == '__main__':
    target_statevector = w_state(num_qubit)
    experiment = f'w-state/{num_qubit}'
    GA(target_statevector=target_statevector,
        num_qubit = num_qubit,
        num_genes = num_genes,
        length_gene = length_gene,
        mutation_rate = mutation_rate,
        cpu_count = cpu_count,
        path = path,
        optimizer = optimizer,
        optimizer2 = optimizer2,
        maxiter = maxiter,
        miniter = miniter,
        threshold = threshold,
        experiment = experiment,
        GPU = GPU)

        
