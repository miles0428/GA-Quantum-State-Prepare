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
    for i,target_statevector in enumerate(target_statevectors):
        experiment = f'benchmark/{i}'
        GA(num_qubit = num_qubits,
            target_statevector = target_statevector,
            num_genes = mp.cpu_count(),
            length_gene = 20,
            mutation_rate = 0.2,
            path = 'data',
            experiment = f'{experiment}/GA',
            optimizer = optimizers.SPSA(maxiter=1000),
            iter = 30,
            threshold = 0.90,
            num_types = 15)
        #initialize
        depths_GA.append(np.load(f'data/{experiment}/GA/best_gene.npy',allow_pickle=True)['depth'])
        circuit = initialize_circuit(num_qubits,target_statevector)
        #get the depth of the circuit
        depth_initialize = circuit.depth()
        depths_initialize.append(depth_initialize)
        #save the depth
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
