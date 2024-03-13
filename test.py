import sympy
import numpy as np
import cirq
from cirq.contrib.svg import SVGCircuit

def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rz(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def generate():
    # PARAMS
    n_qubits = 3
    
    # Number of qubits
    qubits = cirq.GridQubit.rect(1, n_qubits)
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{12})')
    # params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x_theta(0:{n_qubits})')
    # inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()

    # input angles
    circuit += cirq.Circuit(one_qubit_rotation(qubits[0], inputs[0:3])) # inputs[0,1,2]
    # Training
    circuit += cirq.Circuit(one_qubit_rotation(qubits[1], params[0:3])) # params[0,1,2]
    circuit += entangling_layer(qubits)
    circuit += cirq.Circuit(one_qubit_rotation(qubits[0], params[3:6])) # params[3,4,5]
    # measurement
    circuit += cirq.Circuit(cirq.measure(qubits[0], key='A'))
    # measurement
    circuit += cirq.Circuit(cirq.measure(qubits[1], key='B'))
    # retrieve state by Rz, Rx, Rz from qubit 0 to qubit 2 -- params[6,7,8]
    circuit += cirq.Circuit(
                        cirq.rz(params[6])(qubits[2]).controlled_by(qubits[0]),
                        cirq.rx(params[7])(qubits[2]).controlled_by(qubits[0]),
                        cirq.rz(params[8])(qubits[2]).controlled_by(qubits[0])
                            )
    # retrieve state by Rz, Rx, Rz from qubit 1 to qubit 2
    circuit += cirq.Circuit(
                        cirq.rz(params[9])(qubits[2]).controlled_by(qubits[1]),
                        cirq.rx(params[10])(qubits[2]).controlled_by(qubits[1]),
                        cirq.rz(params[11])(qubits[2]).controlled_by(qubits[1])
                            )

    return circuit, list(params), list(inputs)

if __name__ == "__main__":
    circ, _, _ = generate()
    
    print(circ)