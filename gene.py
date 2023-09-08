import qiskit as qk
import numpy as np
import matplotlib.pyplot as plt
from qiskit.algorithms import optimizers

class Gene_Circuit(object):
    '''
    Gene Circuit

    Methods:
        generate_circuit_from_gene: generate a quantum circuit with num_qubit qubits
        circuit_0 ~ circuit_10: quantum circuit connect with gene

    '''
    def __init__(self,gene,num_qubit) -> None:
        '''
        Args:
            gene: a list of 0-10
            num_qubit: number of qubits
        
        object:
            self.gene: a list of 0-10
            self.num_qubit: number of qubits
            self.circuit: a quantum circuit with num_qubit qubits generated from gene
            self.draw: draw the circuit
        '''
        self.gene = gene
        self.num_qubit = num_qubit
        self.circuit = qk.transpile(self.generate_circuit_from_gene())
        self.draw = self.circuit.draw
        self.num_parameters = self.circuit.num_parameters
        self.depth = self.circuit.depth

    def bind_parameters(self, theta : list|np.ndarray) -> qk.QuantumCircuit:
        '''
        bind parameter to the circuit
        Args:
            theta: a list of theta
        Returns:
            bind_circuit: a quantum circuit with parameter binded
        '''
        binded_circuit = self.circuit.bind_parameters({self.circuit.parameters[i]:theta[i] for i in range(len(theta))})
        return binded_circuit

    def generate_circuit_from_gene(self)->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        Args:
            self.gene: a list of 0-10
            self.num_qubit: number of qubits
        Returns:
            circuit: a quantum circuit
        '''
        theta_index = 0
        circuit = qk.QuantumCircuit(self.num_qubit)
        circuit.h(range(self.num_qubit))
        for i in self.gene:
            # newcircuitinfo=eval(f'circuit_{i}({self.num_qubit}, {theta_index})')
            newcircuitinfo = self.__getattribute__(f'circuit_{i}')(theta_index)
            newcircuit = newcircuitinfo[0]
            theta_index = newcircuitinfo[1]
            circuit=circuit.compose(newcircuit)
        #combine two nearby rotation gate into one 
        # circuit.measure_all()
        return circuit

    def circuit_0(self,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        entangled from 2n qubit to 2n-1 qubit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            if i%2==0 and i!=num_qubit-1:
                circuit.cx(i, i+1)
        return (circuit, theta_index)

    def circuit_1(self,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        entangled from 2n-1 qubit to 2n qubit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            if i%2==1 and i!=num_qubit-1:
                circuit.cx(i, i+1)
        return (circuit, theta_index)

    def circuit_2(self ,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        entangled from n qubit to n+2 qubit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            if i<num_qubit-2:
                circuit.cx(i, i+2)
        return (circuit, theta_index)

    def circuit_3(self,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        add a x gate to the each qubit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            circuit.x(i)
        return (circuit, theta_index)

    def circuit_4(self,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        add a ry gate to even qubit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            if i%2==0:
                circuit.ry(qk.circuit.Parameter(f'theta_{theta_index}'), i)
                theta_index+=1
        return (circuit, theta_index)

    def circuit_5(self,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        add a ry gate to odd qubit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            if i%2==1:
                circuit.ry(qk.circuit.Parameter(f'theta_{theta_index}'), i)
                theta_index+=1
        return (circuit, theta_index)

    def circuit_6(self,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        empty circuit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        return (circuit, theta_index)

    def circuit_7(self,theta_index :int )->qk.QuantumCircuit:
        '''    
        Generate a quantum circuit with num_qubit qubits
        entangled from 2n-1 qubit to 2n qubit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta

        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            if i%2==0 and i!=num_qubit-1:
                circuit.cx(i+1, i)
        return (circuit, theta_index)

    def circuit_8(self,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        entangled from 2n qubit to 2n-1 qubit   
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta
        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            if i%2==1 and i!=num_qubit-1:
                circuit.cx(i+1, i)
        return (circuit, theta_index)

    def circuit_9(self,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        ry for prime number qubit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta
        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            if i in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
                circuit.ry(qk.circuit.Parameter(f'theta_{theta_index}'), i)
                theta_index+=1
        return (circuit, theta_index)

    def circuit_10(self,theta_index :int )->qk.QuantumCircuit:
        '''
        Generate a quantum circuit with num_qubit qubits
        ry for not prime number qubit
        Args:
            self.num_qubit: number of qubits
            theta_index: index of the last theta
        Returns:
            (circuit, theta_index): a quantum circuit and the index of the last theta
        '''
        num_qubit = self.num_qubit
        circuit = qk.QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            if i not in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
                circuit.ry(qk.circuit.Parameter(f'theta_{theta_index}'), i)
                theta_index+=1        
        return (circuit, theta_index)
    

if __name__ == "__main__":
    '''
    test code
    '''
    gene=[0,1,2,3,4,5,6,7,8,9,10]
    num_qubit=4
    gene_circuit = Gene_Circuit(gene,num_qubit)
    print(gene_circuit.draw())
    # print(gene_circuit.circuit.data)

    print(gene_circuit.circuit.data[28])