'''
this file is used to compare the final depth of the circuit between GA and Qiskit's initialize algorithm
'''
import qiskit as qk
import numpy as np
import matplotlib.pyplot as plt
from GA import GA
import transform
from qiskit_algorithms import optimizers
import multiprocessing as mp
from qiskit import qpy
import os

mp.set_start_method('spawn',True)

def initialize_circuit(num_qubits,statevector) -> qk.QuantumCircuit:
    basis_gates=["u3","u2","u1","cx","u0","u","p","x","y","z","h","s",
                    "sdg","t","tdg","rx","ry","rz","sx","sxdg","cz","cy",
                    "swap"]
    qubit=qk.QuantumRegister(num_qubits)
    circuit=qk.QuantumCircuit(qubit)
    circuit.initialize(statevector,qubit)
    circuit_n=qk.compiler.transpile(circuit,basis_gates=basis_gates)
    return circuit_n

def main():
    seed = 10292
    np.random.seed(seed)
    num_qubits = 5
    target_statevectors=[]
    for i in range(40):
        target_statevector = transform.normalize_state_vector(np.random.rand(2**num_qubits))
        target_statevectors.append(target_statevector)
    depths_initialize = []
    depths_GA = []
    num_genes = mp.cpu_count()
    num_qubit = 5
    length_gene = 70
    mutation_rate = 0.1
    cpu_count = mp.cpu_count()//2
    optimizer = optimizers.SPSA(maxiter=1500)
    optimizer2 = optimizers.COBYLA(maxiter=1500)
    maxiter = 100
    miniter = 10
    threshold = 0.6
    cpu_count = mp.cpu_count()//2
    GPU = False
    for i,target_statevector in enumerate(target_statevectors):
        experiment = f'benchmark/{i}'
        GA( target_statevector = target_statevector,
            num_qubit = num_qubit,
            num_genes = num_genes,
            length_gene = length_gene,
            mutation_rate = mutation_rate,
            cpu_count = cpu_count,
            path = 'data',
            experiment = f'{experiment}/GA',
            optimizer = optimizer,
            optimizer2 = optimizer2,
            maxiter = maxiter,
            miniter = miniter,
            threshold = threshold,
            GPU = GPU
            )


        #initialize
        depths_GA.append(np.load(f'data/{experiment}/GA/best_gene.npy',allow_pickle=True).item()['depth'])
        circuit = initialize_circuit(num_qubits,target_statevector)
        #get the depth of the circuit
        depth_initialize = circuit.depth()
        depths_initialize.append(depth_initialize)
        #save the depth
        os.makedirs(f'data/{experiment}/qiskit',exist_ok=True)
        np.save(f'data/{experiment}/qiskit/initialize.npy',depth_initialize)
        #save the target statevector
        np.save(f'data/{experiment}/qiskit/target_statevector.npy',target_statevector)
        #save the circuit
        with open(f'data/{experiment}/qiskit/initialize.qpy','wb') as f:
            qpy.dump(circuit,f)
    plot_bechmark(depths_GA,depths_initialize)


def plot_bechmark(depths_GA,depths_qiskit):
    plt.clf()
    plt.xlabel('depth_Qiskit')
    plt.ylabel('depth_GA')
    plt.scatter(depths_qiskit,depths_GA)
    #x=y
    plt.plot(np.arange(0,max(max(depths_GA),max(depths_qiskit))),np.arange(0,max(max(depths_GA),max(depths_qiskit))))
    plt.savefig('benchmark/benchmark.png')

if __name__ == '__main__':
    main()