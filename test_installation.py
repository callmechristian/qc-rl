import tensorflow as tf
import tensorflow_quantum as tfq
import cirq

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Test TensorFlow Quantum installation
print("TensorFlow Quantum version:", tfq.__version__)

# Test REPO installation
from PQC import *
from cirq.contrib.svg import SVGCircuit

n_qubits, n_layers = 3, 1
qubits = cirq.GridQubit.rect(1, n_qubits)
circuit, _, _ = generate_circuit(qubits, n_layers)
SVGCircuit(circuit)

print("Cirq version:", cirq.__version__)
print("Circuit generated successfully! -- you're all set!")