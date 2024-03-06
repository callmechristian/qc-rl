import numpy as np
from abc import ABC

class Environment(ABC):
    env_name = None
    max_steps = None
    n_qubits = None
    n_layers = None
    n_actions = None
    state_bounds = None
    gym = False
    
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
        """
        Calculate the observables for the CustomEnvironment.

        Args:
            ops (list): A list of operators representing the observables.

        Raises:
            NotImplementedError: This method is not implemented for CustomEnvironment.
                Please define the observables_func method in your custom environment.

        Returns:
            Observable products vector.
        """
        raise NotImplementedError("Observables computation not set for Environment! Please define it!")

class CustomEnvironment(Environment):
    """
    CustomEnvironment is a custom environment class that extends the base Environment class.

    Args:
        Environment (ABC): The base abstract class for environments.

    Raises:
        NotImplementedError: If the required attributes are not set for the custom environment.

    Returns:
        None
    """
    env_name = None
    max_steps = None
    n_qubits = None
    n_layers = None # Number of layers in the PQC
    n_actions = None
    state_bounds = None
    gym = False
    
    def __init__(self, env_name: str, max_steps: int, n_qubits: int, n_layers: int, n_actions: int, state_bounds, gym: bool = False) -> None:
        super().__init__()
        self.env_name = env_name
        self.max_steps = max_steps
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.state_bounds = state_bounds
        self.gym = gym
        self.check()

        
    
    def make(self):
        """
        This method is responsible for creating the custom environment.
        
        Raises:
            NotImplementedError: If the method is not implemented in the custom environment.
        """
        raise NotImplementedError("CustomEnvironment make not implemented! Please define the make method in your custom environment.")
    def step(self, action):
        """
        Executes a single step in the environment.

        Args:
            action: The action to take in the environment.

        Raises:
            NotImplementedError: If the `step` method is not implemented in the custom environment.

        Returns:
            The observation, reward, done flag, and additional information.
        """
        raise NotImplementedError("step not implemented for CustomEnvironment! Please define step method in your custom environment.")
    def reset(self):
        """
        Resets the environment to its initial state.
        
        Raises:
            NotImplementedError: If the reset method is not implemented for the custom environment.
        """
        raise NotImplementedError("reset not implemented for CustomEnvironment! Please define reset method in your custom environment.")
    def close(self):
        """
        Closes the environment and performs any necessary cleanup.

        Raises:
            NotImplementedError: If the close method is not implemented for the custom environment.
        """
        raise NotImplementedError("close not implemented for CustomEnvironment! Please define close method in your custom environment.")
    def render(self):
        """
        Renders the current state of the environment.

        Raises:
            NotImplementedError: If the render method is not implemented for the custom environment.
        """
        raise NotImplementedError("render not implemented for CustomEnvironment! Please define render method in your custom environment.")
    def seed(self, seed):
        """
        Set the seed for the environment's random number generator.
        
        Parameters:
            seed (int): The seed value to set.
        
        Raises:
            NotImplementedError: If the `seed` method is not implemented for the custom environment.
        """
        raise NotImplementedError("seed not implemented for CustomEnvironment! Please define seed method in your custom environment.")
    
    def check(self):
        """
        Checks if all the required attributes of the CustomEnvironment are set.
        
        Raises:
            NotImplementedError: If any of the required attributes are not set.
        """
        
        if self.env_name is None:
            raise NotImplementedError("env_name not set for CustomEnvironment!")
        if self.max_steps is None:
            raise NotImplementedError("max_steps not set for CustomEnvironment!")
        if self.n_qubits is None:
            raise NotImplementedError("n_qubits not set for CustomEnvironment!")
        if self.n_layers is None:
            raise NotImplementedError("n_layers not set for CustomEnvironment!")
        if self.n_actions is None:
            raise NotImplementedError("n_actions not set for CustomEnvironment!")
        if self.state_bounds is None:
            raise NotImplementedError("state_bounds not set for CustomEnvironment!")

class AcroBot(Environment):
    env_name="Acrobot-v1"
    max_steps=500
    n_qubits = 6
    n_layers = 8 # Number of layers in the PQC
    n_actions = 3
    state_bounds = np.array([0.3,0.7,0.3,0.7,12,28])
    # https://www.gymlibrary.dev/environments/classic_control/acrobot/
    gym = True    
    
    def observables_func(ops):
        return [ops[0]*ops[2]*ops[4], ops[0]*ops[1]*ops[2]*ops[3]*ops[4]*ops[5], ops[1]*ops[3]*ops[5]]

class CartPole(Environment):
    env_name="CartPole-v1"
    max_steps=500
    n_qubits = 4
    n_layers = 4 # Number of layers in the PQC
    n_actions = 2
    state_bounds = np.array([4.8, 1, 0.418, 1])
    # https://www.gymlibrary.dev/environments/classic_control/cart_pole/
    gym = True
    
    # learning rates 0.001 0.001 0.1
    def observables_func(ops):
        return [ops[0]*ops[1], ops[2]*ops[3]]

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
    
    gym = True
    def observables_func(ops):
        return [ops[0]*ops[1], ops[0]*ops[1]*ops[2], ops[0]*ops[1], -ops[0]*ops[1]]
    
class MountainCar(Environment):
    env_name="MountainCar-v0"
    max_steps=200
    n_qubits = 2
    n_layers = 2 # Number of layers in the PQC
    n_actions = 3
    state_bounds = np.array([1, 1])
    # https://www.gymlibrary.dev/environments/classic_control/mountain_car/
    gym = True    
    
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
    gym = True
    
    def observables_func(ops):
        raise NotImplementedError("Observables not implemented for LunarLander")