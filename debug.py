import matplotlib.pyplot as plt
from train import *
import numpy as np

# ==============================================
# HYPERPARAMETERS ==============================
realtime_render = False
batch_size = 10
n_episodes = 2000
# EXPORT SETTINGS ==============================
note = ""
export_gif = True
# ENVIRONMENT SETTINGS =========================
env_type = Environments.CartPole
method = TrainMethod.DeepQLearning
reward_target = 500.0
# ==============================================

# TRAIN
history, model, env, best_model = train(reward_target=reward_target, realtime_render=realtime_render, batch_size=batch_size, env_type=env_type, method=method, n_episodes=n_episodes)

# EXPORT
export(history, env_type, best_model, method, episodes=n_episodes, note=note, export_gif=export_gif)