import matplotlib.pyplot as plt
from train import *

history = train(reward_target=100.0, realtime_render=False, batch_size=10, env_type=Environments.CartPole, method=TrainMethod.DeepQLearning)