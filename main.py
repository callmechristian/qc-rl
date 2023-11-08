import matplotlib.pyplot as plt
from train import *

history = train(reward_target=100.0, realtime_render=True, batch_size=1)