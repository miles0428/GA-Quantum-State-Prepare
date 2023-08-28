import qiskit as qk
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from qiskit.algorithms import optimizers
import os
import random
optimizer = optimizers.SPSA(maxiter=1000)
# optimizer = optimizers.ADAM(maxiter=1000, tol=0.0001)

num_qubit = 4
#create 8 different circuits with num_qubit qubits
theta_index = 0

def circuit_0(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    entangled from 2n qubit to 2n-1 qubit
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        if i%2==0 and i!=num_qubit-1:
            circuit.cx(i, i+1)
    circuit.barrier()
    return (circuit, theta_index)

def circuit_1(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    entangled from 2n-1 qubit to 2n qubit
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        if i%2==1 and i!=num_qubit-1:
            circuit.cx(i, i+1)
    circuit.barrier()
    return (circuit, theta_index)

def circuit_2(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    entangled from n qubit to n+2 qubit
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        if i<num_qubit-2:
            circuit.cx(i, i+2)
    circuit.barrier()
    return (circuit, theta_index)

def circuit_3(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    add a x gate to the each qubit
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        circuit.x(i)
    circuit.barrier()
    return (circuit, theta_index)

def circuit_4(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    add a ry gate to even qubit
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        if i%2==0:
            circuit.ry(qk.circuit.Parameter(f'theta_{theta_index}'), i)
            theta_index+=1
    circuit.barrier()
    return (circuit, theta_index)

def circuit_5(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    add a ry gate to odd qubit
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        if i%2==1:
            circuit.ry(qk.circuit.Parameter(f'theta_{theta_index}'), i)
            theta_index+=1
    circuit.barrier()
    return (circuit, theta_index)

def circuit_6(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    empty circuit
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta

    '''
    circuit = qk.QuantumCircuit(num_qubit)
    circuit.barrier()
    return (circuit, theta_index)

def circuit_7(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''    
    Generate a quantum circuit with num_qubit qubits
    entangled from 2n-1 qubit to 2n qubit
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta

    '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        if i%2==0 and i!=num_qubit-1:
            circuit.cx(i+1, i)
    circuit.barrier()
    return (circuit, theta_index)

def circuit_8(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    entangled from 2n qubit to 2n-1 qubit   
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta
            '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        if i%2==1 and i!=num_qubit-1:
            circuit.cx(i+1, i)
    circuit.barrier()
    return (circuit, theta_index)

def circuit_9(num_qubit : int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    ry for prime number qubit
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta
            '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        if i in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
            circuit.ry(qk.circuit.Parameter(f'theta_{theta_index}'), i)
            theta_index+=1

    circuit.barrier()
    return (circuit, theta_index)

def circuit_10(num_qubit :int ,theta_index :int )->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    ry for not prime number qubit
    
    
    Args:
        num_qubit: number of qubits
    Returns:
        (circuit, theta_index): a quantum circuit and the index of the last theta
            
            '''
    circuit = qk.QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        if i not in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
            circuit.ry(qk.circuit.Parameter(f'theta_{theta_index}'), i)
            theta_index+=1
    
    circuit.barrier()
    return (circuit, theta_index)

def generate_circuit_from_gene(gene : list, num_qubit : int)->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    Args:
        gene: a list of 0-10
        num_qubit: number of qubits
    Returns:
        circuit: a quantum circuit
    '''
    theta_index = 0
    circuit = qk.QuantumCircuit(num_qubit)
    circuit.h(range(num_qubit))
    for i in gene:
        newcircuitinfo=eval(f'circuit_{i}({num_qubit}, {theta_index})')
        newcircuit = newcircuitinfo[0]
        theta_index = newcircuitinfo[1]
        # circuit.draw('mpl', filename='circuit_t.png')
        # newcircuit.draw('mpl', filename='newcircuit.png')
        circuit=circuit.compose(newcircuit)
    circuit.measure_all()
    return circuit


def get_prob_distribution(circuit, theta):
    #get the probability distribution
    #bind the parameter
    circuit = circuit.bind_parameters({circuit.parameters[i]:theta[i] for i in range(len(theta))})
    backend = qk.Aer.get_backend('qasm_simulator')
    #execute the circuit
    job = qk.execute(circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts()
    # print(counts)
    prob_distribution = np.zeros(2**num_qubit)
    for key in counts.keys():
        prob_distribution[int(key, 2)] = counts[key]/1000
    # prob_distribution = np.array([counts.get(key, 0) for key in counts.keys()])/1000
    return prob_distribution

def get_optimized_fidelity(gene:list, num_qubit:int):
    '''
    Get the optimized fidelity of a gene
    Args:
        gene: a list of 0-10
        num_qubit: number of qubits
    Returns:
        fidelity: the optimized fidelity of the gene
        depth: the depth of the circuit
        theta: the optimized theta
    '''
    circuit = generate_circuit_from_gene(gene, num_qubit)
    backend = qk.Aer.get_backend('qasm_simulator')
    #optimize the parameter
    circuit = qk.transpile(circuit, backend)
    #load the target distribution
    target_distribution = np.load('gaussion_avg_7.5_sigma_3.npy')
    target_distribution = target_distribution/np.sum(target_distribution)
    #define the fidelity function
    def get_fidelity(theta):
        #calculate the probability distribution
        prob_distribution = get_prob_distribution(circuit, theta)
        #calculate the fidelity
        fidelity = np.sum(np.sqrt(np.multiply(target_distribution, prob_distribution)))
        return fidelity
    #define the loss function
    def loss(theta):
        prob_distribution = get_prob_distribution(circuit, theta)
        mse = np.sum(np.square(np.subtract(target_distribution, prob_distribution)))
        return mse

    #get number of parameters
    num_parameter = circuit.num_parameters
    #optimize the parameter
    theta = np.random.rand(num_parameter)
    theta = optimizer.minimize(loss, x0=theta)
    #plot the probability distribution compared with the target distribution
    # prob_distribution = get_prob_distribution(circuit, theta.x)
    #clear the plot
    # plt.clf()
    # plt.plot(np.arange(2**num_qubit), target_distribution, label='target_distribution')
    # plt.bar(np.arange(2**num_qubit), prob_distribution, label='prob_distribution')
    # plt.savefig('prob_distribution_1st_version.png')
    # print(theta)
    fidelity=get_fidelity(theta.x)
    depth=circuit.depth()
    return fidelity,depth,theta.x

gene = [0,1,2,3,4,5,6,7,8,9,10]
circuit = generate_circuit_from_gene(gene, num_qubit) 

#calculate the fidelity and depth for each gene
#use multiprocessing to speed up
def get_fidelity_depth(gene):
    num_qubit = 4
    fidelity,depth,theta = get_optimized_fidelity(gene, num_qubit)
    return fidelity,depth,theta

random_gene = np.random.randint(0,11,128*30).reshape(128,30)
# random_gene=np.load('9st_generation/random_gene.npy')

mutation_rate = 0.1

for i in range(30):

    #use multiprocessing to speed up
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(get_fidelity_depth, random_gene)
    #mkdir 1st_generation
    result=np.array(result,dtype=object)
    pool.close()
    os.makedirs(f'ogaussian/data/{i}st_generation',exist_ok=True)
    #save the result
    np.save(f'gaussian/data/{i}st_generation/result.npy', result)
    #save the random gene
    np.save(f'gaussian/data/{i}st_generation/random_gene.npy', random_gene)

    #load the result
    result = np.load(f'gaussian/data/{i}st_generation/result.npy', allow_pickle=True)
    #plot the result
    # print(result)

    # find the gene with the fidelity larger than 0.99
    gene = result[:,0]>0.99
    index=np.array([]).astype(int)
    #get the index of 10 genes with the smallest depth and fidelity larger than 0.99

    for j in np.argsort(result[:,1]):
        if gene[j]:
            index=np.append(index,j)
            if len(index)==10:
                break

    # print(f'index:{index}')
    print(f'depth:{result[index,1]}')
    print(f'fidelity:{result[index,0]}')
    #save the 10 genes
    np.save(f'gaussian/data/{i}st_generation/10_smallest_depth_gene.npy', random_gene[index])
    np.save(f'gaussian/data/{i}st_generation/10_smallest_depth_result.npy', result[index])
    
    parent_gene = random_gene[index]
    #use parent gene to generate child gene
    child_gene = np.zeros((128,30)).astype(int)
    for j in range(128):
        #randomly choose a parent gene
        parent = [np.random.randint(0,len(index)), np.random.randint(0,len(index))]
        while parent[0]==parent[1]:
            parent[1]=np.random.randint(0,len(index))
        #randomly choose a crossover point
        crossover_point = np.random.randint(1,29)
        #generate child gene
        child_gene[j] = np.concatenate((parent_gene[parent[0]][:crossover_point], parent_gene[parent[1]][crossover_point:]))
        #randomly mutate the child gene
        for k in range(30):
            if np.random.rand()<mutation_rate:
                child_gene[j][k]=np.random.randint(0,11)
    
    #randomly generate 10 genes
    child_gene[128-10:] = np.random.randint(0,11,10*30).reshape(10,30)
    #add the 10 genes with the smallest depth
    child_gene[128-10-len(index):128-10] = random_gene[index]
    random_gene = child_gene.astype(int)

        
        







