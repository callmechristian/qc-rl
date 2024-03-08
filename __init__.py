# __init__.py

# This is the initialization file for the qc-rl package.
from . import train
from .train import TrainMethod, EpsilonDecay

print("Imported train module")