import matplotlib.pyplot as plt
from train import *
import numpy as np
import argparse

def train_and_export(reward_target : float, realtime_render : bool, batch_size : int, env_type : Environments.Environment, method : TrainMethod, n_episodes : int):
    print(f"\n{env_type.n_qubits}\n")
    history, model, env = train(reward_target=reward_target, realtime_render=realtime_render, batch_size=batch_size, env_type=env_type, method=method, n_episodes=n_episodes)
    export(history, env_type, model, method, note="_DeepQ_max")
    # model.save("models/CartPole/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_target", type=float, help="Target reward for training", default=500.0)
    parser.add_argument("--realtime_render", action="store_true", help="Enable real-time rendering", default=False)
    parser.add_argument("--batch_size", type=int, help="Training batch size", default=10)
    parser.add_argument("--env_type", type=str, help="Environment type", choices=["CartPole", "AcroBot", "AtariBreakout"], default="CartPole", required=True)
    parser.add_argument("--method", type=str, help="Training method", choices=["DeepQLearning", "PolicyGradient"], default="DeepQLearning", required=True)
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to train for", default=2000)

    args = parser.parse_args()

    if args.method == "DeepQLearning":
        parsed_method = TrainMethod.DeepQLearning
    elif args.method == "PolicyGradient":
        parsed_method = TrainMethod.PolicyGradient
    else:
        raise ValueError("Please provide a valid training method. Options are PolicyGradient and DeepQLearning")
    
    if args.env_type == "CartPole":
        parsed_env = Environments.CartPole
    elif args.env_type == "AcroBot":
        parsed_env = Environments.AcroBot
    elif args.env_type == "AtariBreakout":
        parsed_env = Environments.AtariBreakout
    else:
        raise ValueError("Invalid environment. Choices: CartPole, AcroBot, AtariBreakout")

    train_and_export(args.reward_target, args.realtime_render, args.batch_size, parsed_env, parsed_method, args.n_episodes)