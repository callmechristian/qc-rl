import numpy as np
import matplotlib.pyplot as plt

def softmax(x, tau):
    e_x = np.exp((x - np.max(x)) / tau)
    return e_x / np.sum(e_x)

# Define the range of values
x = np.linspace(-10, 10, 100)

# Define the tunable parameter
tau = 1.0

# Compute the softmax values for each value in the range
y = softmax(x, tau)

# Plot the softmax values
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Softmax(x)')
plt.title('Softmax Function')
plt.grid(True)
plt.show()