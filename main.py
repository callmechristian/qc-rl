import matplotlib.pyplot as plt
from train import *
import numpy as np
import argparse
from data import Environments

def train_and_export(reward_target : float, realtime_render : bool, batch_size : int, env_type : Environments.Environment, method : TrainMethod, n_episodes : int, note : str = "", export_gif : bool = False):
    print(f"\n{env_type.n_qubits}\n")
    history, model, env, best_model = train(reward_target=reward_target, realtime_render=realtime_render, batch_size=batch_size, env_type=env_type, method=method, n_episodes=n_episodes)
    if export_gif: 
        export(history, env_type, model, method, episodes=n_episodes, note=note)
        # model.save("models/CartPole/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-target", type=float, help="Target reward for training", default=500.0)
    parser.add_argument("--realtime-render", action="store_true", help="Enable real-time rendering", default=False)
    parser.add_argument("--batch-size", type=int, help="Training batch size", default=10)
    parser.add_argument("--env-type", type=str, help="Environment type", choices=["CartPole", "AcroBot", "MountainCar", "AtariBreakout"], default="CartPole", required=True)
    parser.add_argument("--method", type=str, help="Training method", choices=["DeepQLearning", "REINFORCE"], default="DeepQLearning", required=True)
    parser.add_argument("--n-episodes", type=int, help="Number of episodes to train for", default=2000)
    parser.add_argument("--note", type=str, help="Note to add to the exported gif file", default="")
    parser.add_argument("--export-gif", action="store_true", help="Export gif of the training process", default=False)

    args = parser.parse_args()

    if args.method == "DeepQLearning":
        parsed_method = TrainMethod.DeepQLearning
    elif args.method == "REINFORCE":
        parsed_method = TrainMethod.REINFORCE
    else:
        raise ValueError("Please provide a valid training method. Options are REINFORCE and DeepQLearning")
    
    if args.env_type == "CartPole":
        parsed_env = Environments.CartPole
    elif args.env_type == "AcroBot":
        parsed_env = Environments.AcroBot
    elif args.env_type == "AtariBreakout":
        parsed_env = Environments.AtariBreakout
    elif args.env_type == "MountainCar":
        parsed_env = Environments.MountainCar
    else:
        raise ValueError("Invalid environment. Choices: CartPole, AcroBot, AtariBreakout")

    train_and_export(args.reward_target, args.realtime_render, args.batch_size, parsed_env, parsed_method, args.n_episodes, args.note, args.export_gif)