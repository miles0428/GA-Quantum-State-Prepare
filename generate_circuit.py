import qiskit as qk
# create 16 parameters

def generate_circuit(num_qubit : int)->qk.QuantumCircuit:
    '''
    Generate a quantum circuit with num_qubit qubits
    Args:
        num_qubit: number of qubits
    Returns:
        circuit: a quantum circuit
        '''
    theta =[qk.circuit.Parameter(f'theta{i}') for i in range(num_qubit*4)]

    #create 4 qubits
    q = qk.QuantumRegister(num_qubit, 'q')
    # create a circuit
    circuit = qk.QuantumCircuit(q)

    # apply Hadamard gate to all qubits
    circuit.h(q)

    # apply rotation gates for each qubit
    for i,o in enumerate(theta):
        circuit.ry(o, q[i%num_qubit])
        if i%num_qubit==num_qubit-1:
            for j in range(num_qubit):
                circuit.cx(q[j], q[(j+1)%num_qubit])
            circuit.barrier()

    # measure all qubits
    circuit.measure_all()
    # circuit.draw('mpl', filename='circuit.png')
    return circuit

generate_circuit(5).draw('mpl', filename='circuit.png')