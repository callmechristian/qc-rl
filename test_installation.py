import tensorflow as tf
import tensorflow_quantum as tfq

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")

# Test TensorFlow Quantum installation
circuit = tfq.convert_to_tensor([cirq.Circuit()])
output = tfq.layers.Expectation()(circuit)
print("TensorFlow Quantum installation is working")
