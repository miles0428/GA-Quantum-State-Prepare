'''
This file is used to implement the genetic algorithm on the quantum circuit for preparing the target statevector

author : Yu-Cheng Chung
email  : ycchung@ntnu.edu.tw
date   : 2023 08 Sep

dependencies:

    gene.py
    qiskit
    numpy
    multiprocessing
    qiskit_algorithms


'''
import qiskit as qk
import numpy as np
import multiprocessing as mp
from qiskit_algorithms import optimizers
import os
from gene import Gene_Circuit
from functools import partial

def get_prob_distribution(circuit : qk.QuantumCircuit, theta : list|np.ndarray, num_qubits :int) -> np.ndarray:
    '''
    Get the probability distribution of a circuit
    Args:
        circuit: a quantum circuit
        theta: a list of theta
        num_qubits: number of qubits
    Returns:
        prob_distribution: the probability distribution of the circuit
    '''
    circuit = circuit.bind_parameters({circuit.parameters[i]:theta[i] for i in range(len(theta))})
    circuit.measure_all()
    backend = qk.Aer.get_backend('qasm_simulator')
    job = qk.execute(circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    prob_distribution = np.zeros(2**num_qubits)
    for key in counts.keys():
        prob_distribution[int(key, 2)] = counts[key]/1000
    return prob_distribution

#define the fidelity function
def get_fidelity(statevector : np.ndarray, target_statevector : np.ndarray) -> float:
    '''
    Get the fidelity of a statevector
    Args:
        statevector: the statevector of the circuit
        target_statevector: the target statevector
    Returns:
        fidelity: the fidelity of the statevector
    '''
    fidelity = np.abs(np.dot(np.conj(np.array(statevector)), target_statevector))**2
    return fidelity

def statevector(Gene : Gene_Circuit, theta : np.ndarray, backend : qk.providers.backend) -> np.ndarray:
    '''
    Get the statevector of a circuit
    Args:
        Gene : Gene_Circuit
        theta: a list of theta
        backend: the backend of the circuit
    Returns:
        statevector: the statevector of the circuit
    '''
    circuit = Gene.bind_parameters(theta)
    job = qk.execute(circuit, backend)
    result = job.result()
    statevector = result.get_statevector()
    return statevector

def get_optimized_fidelity(Gene : Gene_Circuit, target_statevector:np.ndarray ,**kwargs) -> (float, int, np.ndarray):
    '''
    Get the optimized fidelity of a gene
    Args:
        Gene: Gene_Circuit
        target_statevector: the target statevector
    
    kwargs:
        optimizer: the optimizer of the circuit. Default: optimizers.SPSA(maxiter=1000)

    Returns:
        fidelity: the optimized fidelity of the gene
        depth: the depth of the circuit
        theta: the optimized theta
    '''
    backend = qk.Aer.get_backend('statevector_simulator')
    num_parameters = Gene.num_parameters
    theta = np.random.rand(num_parameters)
    try:
        optimizer = kwargs['optimizer']
    except:
        optimizer = optimizers.SPSA(maxiter=1000)
    #define the loss function
    def loss(theta):
        fidelity = get_fidelity(statevector(Gene, theta, backend), target_statevector)
        loss = -fidelity
        return loss
    
    theta = optimizer.minimize(loss, x0=theta)
    #get the optimized probability distribution
    fidelity=get_fidelity(statevector(Gene, theta.x, backend), target_statevector)
    depth=Gene.depth()
    return fidelity,depth,theta.x

def get_fidelity_depth(gene : list, **kwargs ) -> (float, int, np.ndarray):
    '''
    this function is used to get the fidelity and depth of a gene
    Args:
        gene: a list of 0-10

    kwargs:
        num_qubit: number of qubits
        target_statevector: the target statevector
        optimizer: the optimizer of the circuit. Default: optimizers.SPSA(maxiter=1000)

    Returns:
        fidelity: the fidelity of the gene
        depth: the depth of the circuit
        theta: the optimized theta
    '''
    try:
        num_qubit = kwargs['num_qubit']
    except:
        raise Exception('num_qubit is not defined')
    try:
        target_statevector = kwargs['target_statevector']
    except:
        raise Exception('target_statevector is not defined')
    try:
        optimizer = kwargs['optimizer']
    except:
        optimizer = optimizers.SPSA(maxiter=1000)
    
    Gene = Gene_Circuit(gene, num_qubit)
    # print(gene)
    fidelity,depth,theta = get_optimized_fidelity(Gene, target_statevector,optimizer=optimizer)
    # print(theta)
    # print(statevector(Gene, theta, qk.Aer.get_backend('statevector_simulator')), target_statevector)
    # print(fidelity)
    return fidelity,depth,theta

def get_index(result : np.ndarray) -> np.ndarray:
    ii = 0 #the fidelity threshold
    while (True and ii<100):
        # find the gene with the fidelity larger than 0.99
        gene = result[:,0]>0.99 - 0.01*ii
        ii += 1
        if np.sum(gene)>=6:
            break
        elif 0.99 - 0.01*ii<0.9:
            if np.sum(gene)>3:
                break
            elif np.sum(gene)>0:
                #randomly choose 2 genes
                gene[np.random.randint(0,len(gene))]=True
                gene[np.random.randint(0,len(gene))]=True
                break
            else:
                raise Exception('No gene with fidelity larger than 0.90')
    index=np.array([]).astype(int)
    #get the index of 10 genes with the smallest depth and fidelity larger than 0.99
    for j in np.argsort(result[:,1]):
        if gene[j]:
            index=np.append(index,j)
            if len(index)==10:
                break
    return index

def best_gene(target_statevector:np.ndarray,result:np.ndarray,index:np.ndarray) -> dict:
    '''
    this function is used to get the best gene
    Args:
        target_statevector: the target statevector
        result: the result of the genetic algorithm
        index: the index of the 10 genes with the smallest depth and fidelity larger than 0.99
    Returns:
        dict_best_gene: the best gene
    '''
    dict_best_gene = {'target':target_statevector,
                    'gene':result[index[0],2],
                    'depth':result[index[0],1],
                    'fidelity':result[index[0],0],
                    'theta':result[index[0],2]}
    return dict_best_gene

def get_parent_gene(random_gene : np.ndarray, index : np.ndarray) -> np.ndarray:
    return random_gene[index]

def get_child_gene(random_gene:np.ndarray,parent_gene : np.ndarray,index :np.ndarray ,kwargs:dict) -> np.ndarray:
    '''
    this function is used to generate child gene
    Args:
        random_gene: the random gene
        parent_gene: the parent gene
        index: the index of the 10 genes with the smallest depth and fidelity larger than 0.99
    Returns:
        child_gene: the child gene
    '''
    num_genes = kwargs['num_genes']
    length_gene = kwargs['length_gene']
    mutation_rate = kwargs['mutation_rate']
    child_gene = np.zeros((num_genes,length_gene)).astype(int)
    for j in range(num_genes):
        #randomly choose a parent gene
        parent = [np.random.randint(0,len(index)), np.random.randint(0,len(index))]
        while parent[0]==parent[1]:
            parent[1]=np.random.randint(0,len(index))
        #randomly choose a crossover point
        crossover_point = np.random.randint(1,length_gene-1)
        #generate child gene
        child_gene[j] = np.concatenate((parent_gene[parent[0]][:crossover_point], parent_gene[parent[1]][crossover_point:]))
        #randomly mutate the child gene
        for k in range(length_gene):
            if np.random.rand()<mutation_rate:
                child_gene[j][k]=np.random.randint(0,11)
    #randomly generate 10% genes
    child_gene[num_genes-int(num_genes/10):] = np.random.randint(0,11,int(num_genes/10)*length_gene).reshape(int(num_genes/10),length_gene)
    #add the 10 genes with the smallest depth
    child_gene[num_genes-int(num_genes/10)-len(index):num_genes-int(num_genes/10)] = random_gene[index]
    return child_gene.astype(int)

#rewrite the GA function
def GA(target_statevector : np.ndarray ,num_qubit : int ,**kwargs):
    '''
    this function is used to implement the genetic algorithm on the quantum circuit for preparing the target statevector
    Args:
        target_statevector: the target statevector
        num_qubit: number of qubits
    kwargs:
        num_genes: number of genes. Default: 20
        length_gene: length of gene. Default: 10
        mutation_rate: the mutation rate. Default: 0.1
        cpu_count: the number of cpu used. Default: mp.cpu_count()
        path: the path to save the result. Default: data
        expriement: the name of the expriement. Default: test
        optimizer: the optimizer of the circuit. Default: optimizers.SPSA(maxiter=1000)
        iter: the number of iteration. Default: 30
    Returns:
        None
    '''
    kwargs_default = {'num_genes':20,
               'length_gene':10,
               'mutation_rate':0.1,
               'cpu_count':mp.cpu_count(),
               'path':'data',
               'expriement':'test',
               'optimizer':optimizers.SPSA(maxiter=1000),
               'iter':30}
    for key in kwargs_default.keys():
        if key not in kwargs.keys():
            kwargs[key] = kwargs_default[key]
    num_genes = kwargs['num_genes']
    length_gene = kwargs['length_gene']
    cpu_count = kwargs['cpu_count']
    path = kwargs['path']
    expriement = kwargs['expriement']
    optimizer = kwargs['optimizer']
    iter = kwargs['iter']
    random_gene = np.random.randint(0,11,num_genes*length_gene).reshape(num_genes,length_gene)
    partial_get_fidelity_depth = partial(get_fidelity_depth,
                                         num_qubit=num_qubit, 
                                         target_statevector=target_statevector, 
                                         optimizer=optimizer)
    os.makedirs(path,exist_ok=True)
    os.makedirs(f'{path}/{expriement}',exist_ok=True)
    np.save(f'{path}/{expriement}/target_statevector.npy', target_statevector)
    for i in range(iter):
        #check if the data exist
        if os.path.exists(f'{path}/{expriement}/{i}st_generation/result.npy'):
            if os.path.exists(f'{path}/{expriement}/{i+1}st_generation/random_gene.npy'):
                continue
            else:
                result = np.load(f'{path}/{expriement}/{i}st_generation/result.npy', allow_pickle=True)
                random_gene = np.load(f'{path}/{expriement}/{i}st_generation/random_gene.npy', allow_pickle=True)
                index=get_index(result)
                print(f'depth:{result[index,1]}')
                print(f'fidelity:{result[index,0]}')
                #save the 10 genes
                np.save(f'{path}/{expriement}/{i}st_generation/10_smallest_depth_gene.npy', random_gene[index])
                np.save(f'{path}/{expriement}/{i}st_generation/10_smallest_depth_result.npy', result[index])
                #save the best gene
                np.save(f'{path}/{expriement}/best_gene.npy', best_gene(target_statevector,result,index))
                random_gene = get_child_gene(random_gene,get_parent_gene(random_gene,index),index,kwargs)
                print(f'generation {i} finished')
                continue

        #use multiprocessing to speed up
        pool = mp.Pool(cpu_count)
        result = pool.map(partial_get_fidelity_depth, random_gene)
        #mkdir ist_generation
        result=np.array(result,dtype=object)
        pool.close()
        os.makedirs(f'{path}/{expriement}/{i}st_generation',exist_ok=True)
        #save the result
        np.save(f'{path}/{expriement}/{i}st_generation/result.npy', result)
        #save the random gene
        np.save(f'{path}/{expriement}/{i}st_generation/random_gene.npy', random_gene)
        index=get_index(result)
        print(f'depth:{result[index,1]}')
        print(f'fidelity:{result[index,0]}')
        #save the 10 genes
        np.save(f'{path}/{expriement}/{i}st_generation/10_smallest_depth_gene.npy', random_gene[index])
        np.save(f'{path}/{expriement}/{i}st_generation/10_smallest_depth_result.npy', result[index])
        #save the best gene
        np.save(f'{path}/{expriement}/best_gene.npy', best_gene(target_statevector,result,index))
        random_gene = get_child_gene(random_gene,get_parent_gene(random_gene,index),index,kwargs)
        print(f'generation {i} finished')


if __name__ == '__main__':
    GA(np.array([1,0,0,0,0,0,0,-1])/np.sqrt(2), 3, iter=3, expriement='test')