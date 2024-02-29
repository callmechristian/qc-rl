import numpy as np
from abc import ABC

class Environment(ABC):
    env_name = None
    max_steps = None
    n_qubits = None
    n_layers = None
    n_actions = None
    state_bounds = None
    
    """
    Compute the observables from the outputs.

    Args:
        ops (list): The outputs of the PQC.

    Returns:
        list: The computed observables.

    Raises:
        NotImplementedError: If observables are not implemented for the specific environment.
    """
    def observables_func(self, ops):
        return ops

class AcroBot(Environment):
    env_name="Acrobot-v1"
    max_steps=500
    n_qubits = 6
    n_layers = 8 # Number of layers in the PQC
    n_actions = 3
    state_bounds = np.array([0.3,0.7,0.3,0.7,12,28])
    # https://www.gymlibrary.dev/environments/classic_control/acrobot/
    
    def observables_func(ops):
        return [ops[0]*ops[2]*ops[4], ops[0]*ops[1]*ops[2]*ops[3]*ops[4]*ops[5], ops[1]*ops[3]*ops[5]]

class CartPole(Environment):
    env_name="CartPole-v1"
    max_steps=500
    n_qubits = 4
    n_layers = 5 # Number of layers in the PQC
    n_actions = 2
    state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
    # https://www.gymlibrary.dev/environments/classic_control/cart_pole/
    # learning rates 0.001 0.001 0.1
    def observables_func(ops):
        return [ops[0]*ops[2], ops[2]*ops[3]]

class AtariBreakout(Environment):
    env_name="ALE/Breakout-v5"
    max_steps=500
    n_qubits = 3
    n_layers = 5 # Number of layers in the PQC
    n_actions = 4 # restricted from 16
    state_bounds = 255
    # https://www.gymlibrary.dev/environments/atari/breakout/
    # ACTIONS: 0 (NOOP), 1 (FIRE), 2 (RIGHT), 3 (LEFT)
    # [ball_pos, player_pos, remaining_lives]
    def observables_func(ops):
        return [ops[0]*ops[1], ops[0]*ops[1]*ops[2], ops[0]*ops[1], -ops[0]*ops[1]]
    
class MountainCar(Environment):
    env_name="MountainCar-v0"
    max_steps=200
    n_qubits = 2
    n_layers = 5 # Number of layers in the PQC
    n_actions = 3
    state_bounds = np.array([1, 1])
    # https://www.gymlibrary.dev/environments/classic_control/mountain_car/
    
    def observables_func(ops):
        return [ops[0], ops[0]*ops[1], ops[1]]
    
class LunarLander(Environment):
    env_name="LunarLander-v2"
    max_steps=500
    n_qubits = 8
    n_layers = 5 # Number of layers in the PQC
    n_actions = 4
    state_bounds = None
    # https://www.gymlibrary.dev/environments/box2d/lunar_lander/
    
    def observables_func(ops):
        raise NotImplementedError("Observables not implemented for LunarLander")