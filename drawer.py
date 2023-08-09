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
    # circuit.barrier()
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
    # circuit.barrier()
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
    # circuit.barrier()
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
    # circuit.barrier()
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
    # circuit.barrier()
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
    # circuit.barrier()
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
    # circuit.barrier()
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
    # circuit.barrier()
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
    # circuit.barrier()
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

    # circuit.barrier()
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
    
    # circuit.barrier()
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

def draw_circuit_from_gene(gene : list, num_qubit : int,filename:str):
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
        circuit=circuit.compose(newcircuit)

    circuit.measure_all()
    circuit.draw('mpl', filename=filename)
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

def draw_prob_distribution(gene,theta,num_qubit,filename:str):
    circuit = generate_circuit_from_gene(gene, num_qubit)
    prob_distribution = get_prob_distribution(circuit, theta)
    target_distribution = np.load('gaussion_avg_7_sigma_4.npy')
    plt.clf()
    plt.bar(range(2**num_qubit), prob_distribution)
    plt.plot(np.arange(2**num_qubit), target_distribution, label='target_distribution')
    plt.ylim(0,0.4)    
    plt.savefig(filename)

def load_results_from_file(filename:str):
    result = np.load(filename, allow_pickle=True)
    return result

def load_genes_from_file(filename:str):
    gene = np.load(filename)
    return gene


'''
results structure:
    result = [[fidelity, depth, theta], [fidelity, depth, theta], ...]
    
genes structure:
    genes = [[gene], [gene], ...]

'''

#get all the 10_smallest_depth_gene.npy in all the files in the folder
#for each folder in the data folder
#get the generation number
#load the 10_smallest_depth_gene.npy

#{generation_number: [gene, gene, gene, ...]}
genes = {}
for i in range(30) :
    filename = f'data/{i}st_generation/10_smallest_depth_gene.npy'
    gene = load_genes_from_file(filename)
    genes[i] = gene

#load the results
results = {}
for i in range(30) :
    filename = f'data/{i}st_generation/10_smallest_depth_result.npy'
    result = load_results_from_file(filename)
    results[i] = result

#draw fidelity change with generation
fidelity_change = []
for i in range(30):
    fidelity_change.append(sum(results[i][:,0])/10)
plt.clf()
plt.plot(range(30), fidelity_change)
plt.savefig('fidelity_change.png')

#draw depth change with generation
depth_change = []
for i in range(30):
    depth_change.append(sum(results[i][:,1])/10)
plt.clf()
plt.plot(range(30), depth_change)
plt.savefig('depth_change.png')

#draw the smallest depth gene for each generation
for i in range(30):
    draw_circuit_from_gene(genes[i][0], num_qubit, f'generation_{i}_smallest_depth_gene.png')

#draw prob distribution for the smallest depth gene for each generation
for i in range(30):
    draw_prob_distribution(genes[i][0], results[i][0,2], num_qubit, f'generation_{i}_smallest_depth_gene_prob_distribution.png')
