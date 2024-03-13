from Environments import CustomEnvironment

class QubitEnvironment(CustomEnvironment):
    def __init__(self, **kwargs):
        super(QubitEnvironment, self).__init__(**kwargs)

    def step(self, action):
        # Do something
        return self.state, self.reward, self.done, self.info

    def reset(self):
        # Do something
        return self.state

    def render(self, mode='human'):
        # Do something
        pass

    def close(self):
        # Do something
        pass

    def seed(self, seed=None):
        # Do something
        pass

    def get_action_space(self):
        # Do something
        pass

    def get_observation_space(self):
        # Do something
        pass