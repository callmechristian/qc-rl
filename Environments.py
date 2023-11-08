import numpy as np

class AcroBot:
    env_name="Acrobot-v1"
    max_steps=500

    n_qubits = 6
    n_layers = 5 # Number of layers in the PQC
    n_actions = 3

    state_bounds = np.array([0,1,0,1,2,2])
    # https://www.gymlibrary.dev/environments/classic_control/acrobot/

class CartPole:
    env_name="CartPole-v1"
    max_steps=500

    n_qubits = 4
    n_layers = 5 # Number of layers in the PQC
    n_actions = 2

    state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
    # https://www.gymlibrary.dev/environments/classic_control/cart_pole/