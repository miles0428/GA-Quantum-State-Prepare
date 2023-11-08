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
import transform
mp.set_start_method('spawn',True)

def get_prob_distribution(circuit : qk.QuantumCircuit, theta : list|np.ndarray, method : str = 'qasm') -> np.ndarray:
    '''
    Get the probability distribution of a circuit
    Args:
        circuit: a quantum circuit
        theta: a list of theta
        method: the method to get the probability distribution. Default: 'qasm'
    Returns:
        prob_distribution: the probability distribution of the circuit
    '''
    circuit = circuit.bind_parameters({circuit.parameters[i]:theta[i] for i in range(len(theta))})
    num_qubits = circuit.num_qubits
    if method == 'qasm':
        circuit.measure_all()
        backend = qk.Aer.get_backend('qasm_simulator')
        job = qk.execute(circuit, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        prob_distribution = np.zeros(2**num_qubits)
        for key in counts.keys():
            prob_distribution[int(key, 2)] = counts[key]/1000
    elif method == 'statevector':
        backend = qk.Aer.get_backend('statevector_simulator')
        job = qk.execute(circuit, backend)
        result = job.result()
        statevector = result.get_statevector()
        prob_distribution = transform.statevector2prob(statevector)
    else:
        raise Exception('method should be qasm or statevector')
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

def _statevector(Gene : Gene_Circuit, theta : np.ndarray, backend : qk.providers.backend) -> np.ndarray:
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

def _get_optimized_fidelity(Gene : Gene_Circuit, target_statevector:np.ndarray ,**kwargs) -> (float, int, np.ndarray):
    '''
    Get the optimized fidelity of a gene
    Args:
        Gene: Gene_Circuit
        target_statevector: the target statevector
    
    kwargs:
        optimizer: the optimizer of the circuit. Default: optimizers.SPSA(maxiter=1000)
        GPU: if the computer have an avaliable gpu. Default: False

    Returns:
        fidelity: the optimized fidelity of the gene
        depth: the depth of the circuit
        theta: the optimized theta
    '''
    if kwargs['GPU']:
        backend = qk.Aer.get_backend('statevector_simulator')
        backend.set_options(device='GPU')
    else:
        backend = qk.Aer.get_backend('statevector_simulator')

    num_parameters = Gene.num_parameters
    theta = np.random.rand(num_parameters)
    try:
        optimizer = kwargs['optimizer']
    except:
        optimizer = optimizers.SPSA(maxiter=1000)
    #define the loss function
    def loss(theta):
        fidelity = get_fidelity(_statevector(Gene, theta, backend), target_statevector)
        loss = -fidelity
        return loss
    
    theta = optimizer.minimize(loss, x0=theta)
    #get the optimized probability distribution
    fidelity=get_fidelity(_statevector(Gene, theta.x, backend), target_statevector)
    depth=Gene.depth()
    print(fidelity,depth,theta.x)
    return fidelity,depth,theta.x


def _get_fidelity_depth(gene : list, **kwargs ) -> (float, int, np.ndarray):
    '''
    this function is used to get the fidelity and depth of a gene
    Args:
        gene: a array with shape (num_qubit, length_gene) with element called G_ij

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
    fidelity,depth,theta = _get_optimized_fidelity(Gene, target_statevector,optimizer=optimizer,GPU=kwargs['GPU'])
    # print(theta)
    # print(statevector(Gene, theta, qk.Aer.get_backend('statevector_simulator')), target_statevector)
    # print(fidelity)
    return fidelity,depth,theta


def _get_index(result : np.ndarray,threshold :float = 0.9) -> np.ndarray:
    '''
    get the index of 10 genes with the smallest depth and fidelity larger than threshold
    if there is less than 3 genes with fidelity larger than threshold, randomly choose and add 2 genes
    if there is no gene with fidelity larger than threshold, raise an exception
    Args:
        result: the result of the genetic algorithm
        threshold: the threshold of the fidelity
    Returns:
        index: the index of 10 genes with the smallest depth and fidelity larger than threshold
    Raises:
        ValueError: if threshold is not between 0 and 1
        Exception: if there is no gene with fidelity larger than threshold
    '''
    if not(0<threshold<1):
        raise ValueError('threshold should be between 0 and 1')
    
    ii = 0 #the fidelity threshold
    while (True and ii<100):
        # find the gene with the fidelity larger than 0.99
        gene = result[:,0]>0.99 - 0.01*ii
        ii += 1
        if np.sum(gene)>=6:
            break
        elif 0.99 - 0.01*ii<threshold:
            if np.sum(gene)>3:
                break
            elif np.sum(gene)>0:
                #randomly choose 2 genes
                gene[np.random.randint(0,len(gene))]=True
                gene[np.random.randint(0,len(gene))]=True
                break
            else:
                raise Exception('No gene with fidelity larger than threshold')
    index=np.array([]).astype(int)
    #get the index of 10 genes with the smallest depth and fidelity larger than 0.99
    for j in np.argsort(result[:,1]):
        if gene[j]:
            index=np.append(index,j)
            if len(index)==10:
                break
    return index


def _best_gene(random_genes:np.ndarray,target_statevector:np.ndarray,result:np.ndarray,index:np.ndarray,num_qubit:int) -> dict:
    '''
    this function is used to get the best gene
    Args:
        random_genes: the random genes
        target_statevector: the target statevector
        result: the result of the genetic algorithm
        index: the index of the 10 genes with the smallest depth and fidelity larger than 0.99
        num_qubit: number of qubits
    Returns:
        dict_best_gene: the best gene
    '''
    gene=random_genes[index[0]]
    theta = result[index[0],2]
    dict_best_gene = {'target':target_statevector,
                      'gene':gene,
                      'depth':result[index[0],1],
                      'fidelity':result[index[0],0],
                      'theta':result[index[0],2],
                      'num_qubit':num_qubit,
                      'circuit':Gene_Circuit(gene=gene,num_qubit=num_qubit).bind_parameters(theta)}
    return dict_best_gene


def _get_parent_gene(random_gene : np.ndarray, index : np.ndarray) -> np.ndarray:
    '''
    this function is used to get the parent gene
    Args:
        random_gene: the random gene
        index: the index of the 10 genes with the smallest depth and fidelity larger than threshold
    Returns:
        parent_gene: the parent gene
    '''
    return random_gene[index]


def _get_child_gene(random_gene:np.ndarray,parent_gene : np.ndarray,index :np.ndarray ,kwargs:dict) -> np.ndarray:
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
    num_qubit = kwargs['num_qubit']
    num_types = kwargs['num_types']
    # child_gene = np.zeros((num_genes,length_gene)).astype(int)
    child_gene = np.zeros((num_genes, num_qubit, length_gene, 2)).astype(int)
    for j in range(num_genes):
        #randomly choose a parent gene
        parent = [np.random.randint(0,len(index)), np.random.randint(0,len(index))]
        while parent[0]==parent[1]:
            parent[1]=np.random.randint(0,len(index))
        #randomly choose a crossover point
        crossover_point = np.random.randint(1,length_gene-1)
        #generate child gene
        child_gene[j] = np.concatenate((parent_gene[parent[0]][:,:crossover_point,:], 
                                        parent_gene[parent[1]][:,crossover_point:,:]), 
                                        axis=1)
        #randomly mutate the child gene
        for k in range(num_qubit):
            for l in range(length_gene):
                if np.random.rand()<mutation_rate:
                    child_gene[j][k][l]=(np.random.randint(0,num_types), np.random.randint(0,num_qubit))

    #randomly generate 10% genes
    child_gene[num_genes-int(num_genes/10):] = np.concatenate(
        (np.random.randint(low=0, high=num_types, size=(int(num_genes/10), num_qubit, length_gene, 1)), 
         np.random.randint(low=0, high=num_qubit, size=(int(num_genes/10), num_qubit, length_gene, 1))), 
        axis=3)
    #add the 10 genes with the smallest depth
    child_gene[num_genes-int(num_genes/10)-len(index):num_genes-int(num_genes/10)] = random_gene[index]
    return child_gene.astype(int)


def _save_data(result : np.ndarray, 
              random_gene : np.ndarray,
              generation : int,
              target_statevector : np.ndarray,
              index : np.ndarray,
              num_qubit : int,
              kwargs : dict) -> None:
    '''
    this function is used to save the data
    will save the result, 
                  random gene, 
                  the gene and result of 10 genes with the smallest depth and fidelity larger than threhold, 
                  and the best gene.
    Args:
        result: the result of the genetic algorithm
        random_gene: the random gene
        generation: the generation
        target_statevector: the target statevector
        index: the index of the 10 genes with the smallest depth and fidelity larger than 0.99
        num_qubit: number of qubits
        kwarg: the kwargs of the genetic algorithm
    Returns:
        None
    '''
    path = kwargs['path']
    experiment = kwargs['experiment']
    os.makedirs(f'{path}/{experiment}/{generation}st_generation',exist_ok=True)
    #save the result
    if os.path.exists(f'{path}/{experiment}/{generation}st_generation/result.npy'):
        print(f'{path}/{experiment}/{generation}st_generation/result.npy already exists')
    else:
        np.save(f'{path}/{experiment}/{generation}st_generation/result.npy', result)
    #save the random gene
    if os.path.exists(f'{path}/{experiment}/{generation}st_generation/random_gene.npy'):
        print(f'{path}/{experiment}/{generation}st_generation/random_gene.npy already exists')
    else:
        np.save(f'{path}/{experiment}/{generation}st_generation/random_gene.npy', random_gene)
    np.save(f'{path}/{experiment}/{generation}st_generation/10_smallest_depth_gene.npy', random_gene[index])
    np.save(f'{path}/{experiment}/{generation}st_generation/10_smallest_depth_result.npy', result[index])
    #save the best gene
    np.save(f'{path}/{experiment}/best_gene.npy', _best_gene(random_gene,target_statevector,result,index,num_qubit=num_qubit))


def _gpu_avaliable() -> bool:
    '''
    check if the computer have an avaliable gpu
    Returns:
        gpu_avaliable: if the computer have an avaliable gpu 
    '''
    try:
        backend = qk.Aer.get_backend('statevector_simulator')
        backend.set_options(device='GPU')
        qk.execute(qk.QuantumCircuit(1),backend).result()
        return True
    except :
        return False


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
        experiment: the name of the experiment. Default: test
        optimizer: the optimizer of the circuit. Default: optimizers.SPSA(maxiter=1000)
        maxiter: the number of max iteration. Default: 30
        miniter: the number of min iteration. Default: 10
        threshold: the threshold of the fidelity. Default: 0.90
        num_types: the number of types of the gate. Default: 7
        GPU: if the computer have an avaliable gpu. Default: check if the computer have an avaliable gpu
    Returns:
        None
    '''
    kwargs_default = {'num_genes':20,
                      'length_gene':10,
                      'mutation_rate':0.1,
                      'cpu_count':mp.cpu_count(),               
                      'path':'data',
                      'experiment':'test',
                      'optimizer':optimizers.SPSA(maxiter=1000),
                      'maxiter':30,
                      'miniter':10, 
                      'threshold':0.90,
                      'num_types':7,
                      'GPU':_gpu_avaliable()}
    for key in kwargs_default.keys():
        if key not in kwargs.keys():
            kwargs[key] = kwargs_default[key]
    num_genes = kwargs['num_genes']
    length_gene = kwargs['length_gene']
    cpu_count = kwargs['cpu_count']
    path = kwargs['path']
    experiment = kwargs['experiment']
    optimizer = kwargs['optimizer']
    maxiter = kwargs['maxiter']
    threshold = kwargs['threshold']
    num_types = kwargs['num_types']
    miniter = kwargs['miniter']
    kwargs['num_qubit'] = num_qubit
    #generate random gene
    # random_gene = np.random.randint(0,num_types,num_genes*length_gene).reshape(num_genes,length_gene)
    random_gene = np.concatenate((np.random.randint(low=0, high=num_types, size=(num_genes, num_qubit, length_gene, 1)), 
                                  np.random.randint(low=0, high=num_qubit, size=(num_genes, num_qubit, length_gene, 1))), 
                                  axis=3)
    #create a partial function for multiprocessing
    partial_get_fidelity_depth = partial(_get_fidelity_depth,
                                         num_qubit=num_qubit, 
                                         target_statevector=target_statevector, 
                                         optimizer=optimizer,
                                         GPU = kwargs['GPU'])
    os.makedirs(path,exist_ok=True)
    os.makedirs(f'{path}/{experiment}',exist_ok=True)
    np.save(f'{path}/{experiment}/target_statevector.npy', target_statevector)
    caculate = False
    record_depth = dict()
    for i in range(maxiter):
        #check if the data exist
        if caculate:
            pass
        elif os.path.exists(f'{path}/{experiment}/{i}st_generation/result.npy'):
            if os.path.exists(f'{path}/{experiment}/{i+1}st_generation/random_gene.npy'):
                print(f'generation {i} finished')
                result = np.load(f'{path}/{experiment}/{i}st_generation/result.npy', allow_pickle=True)
                index=_get_index(result,threshold=threshold)
                record_depth[i%10] = np.array(result[index,1])
                continue
            else:
                result = np.load(f'{path}/{experiment}/{i}st_generation/result.npy', allow_pickle=True)
                random_gene = np.load(f'{path}/{experiment}/{i}st_generation/random_gene.npy', allow_pickle=True)
                index=_get_index(result,threshold=threshold)
                record_depth[i%10] = np.array(result[index,1])
                print(f'depth:{result[index,1]}\nfidelity:{result[index,0]}')
                #save the result
                _save_data(result,random_gene,i,target_statevector,index,num_qubit,kwargs)
                parent = _get_parent_gene(random_gene,index)
                random_gene = _get_child_gene(random_gene,parent,index,kwargs)
                print(f'generation {i} finished')
                caculate = True
                continue
        else:
            caculate = True
        os.makedirs(f'{path}/{experiment}/{i}st_generation',exist_ok=True)
        #use multiprocessing to speed up
        pool = mp.Pool(cpu_count)
        print("start multiprocessing")
        result = pool.map(partial_get_fidelity_depth, random_gene)
        #mkdir ist_generation
        result=np.array(result,dtype=object)
        pool.close()
        print("end multiprocessing")
        #save the result
        index=_get_index(result,threshold=threshold)
        print(f'depth:{result[index,1]}\nfidelity:{result[index,0]}')
        _save_data(result,random_gene,i,target_statevector,index,num_qubit,kwargs)
        random_gene = _get_child_gene(random_gene,_get_parent_gene(random_gene,index),index,kwargs)
        print(f'generation {i} finished')
        record_depth[i%10] = np.array(result[index,1])
        if len(record_depth[i%10])<10:
            #fill 1e10 to the array
            record_depth[i%10] = np.concatenate((record_depth[i%10],np.ones(10-len(record_depth[i%10]))*1e10))
        #check convergence
        if i>=miniter-1: #check 10 generations before
            #check the standard deviation of the depth
            r=[]
            for i in record_depth.keys():
                r.append(record_depth[i])
            r=np.array(r).reshape(-1)
            std = np.std(r)
            if std<1e-3:
                break



if __name__ == '__main__':
    GA(np.array([1,0,0,0,0,0,0,-1])/np.sqrt(2), 3, experiment='test')