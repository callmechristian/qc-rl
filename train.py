# package imports
import gym, cirq, os
import numpy as np
from functools import reduce
from PIL import Image
from enum import Enum
import tensorflow as tf

# package implements
from .algorithms import REINFORCE, DeepQLearning
import data.Environments as Environments

class TrainMethod(Enum):
    REINFORCE = 0
    DeepQLearning = 1

episode_reward_history = []

def train(reward_target=500.0, realtime_render: bool = False, batch_size: int = 10, env_type: Environments = Environments.CartPole, method: TrainMethod = TrainMethod.REINFORCE, n_episodes: int = 1000, gamma: float = 0.99, lr_in: float = 0.001, lr_var: float = 0.001, lr_out: float = 0.1):

    if method==TrainMethod.REINFORCE:
        if env_type == Environments.AtariBreakout:
            raise NotImplementedError("Policy Gradient not implemented for atari.")
            return train_policy_gradient_atari(reward_target, realtime_render, batch_size, env_type, n_episodes=n_episodes)
        else:
            return train_policy_gradient(reward_target, realtime_render, batch_size, env_type, n_episodes=n_episodes, gamma=gamma, lr_in=lr_in, lr_var=lr_var, lr_out=lr_out)
    elif method == TrainMethod.DeepQLearning:
        if env_type == Environments.AtariBreakout:
            raise NotImplementedError("Deep Q-Learning not implemented for atari.")
            return train_deepq_atari(reward_target, env_type, batch_size=batch_size, n_episodes=n_episodes)
        else:
            return train_deepq(reward_target, env_type, batch_size=batch_size, n_episodes=n_episodes, lr_in=lr_in, lr_var=lr_var, lr_out=lr_out)
    else:
        raise ValueError("Unrecognized training method! Check the TrainMethod enum for valid methods.")

def train_policy_gradient(reward_target: float, realtime_render: bool, batch_size: int, env_type: Environments.Environment, n_episodes=1000, gamma=0.99, lr_in=0.001, lr_var=0.001, lr_out=0.1):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    observables = env_type.observables_func(ops)
    
    rlagent = REINFORCE(gamma, lr_in, lr_var, lr_out)

    model = rlagent.generate_model_policy(qubits, env_type.n_layers, env_type.n_actions, 0.9, observables)
    
    env = None
    best_model = None
    episode_reward_history = []
    # Start training the agent
    for batch in range(n_episodes // batch_size):
        # Gather episodes
        episodes = rlagent.gather_episodes(env_type.state_bounds, env_type.n_actions, model, batch_size, env_type)
        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([rlagent.compute_returns(ep_rwds, rlagent.gamma) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)
        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        rlagent.reinforce_update(states, id_action_pairs, returns, model, batch_size)

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))

        avg_rewards = np.mean(episode_reward_history[-batch_size:])

        print('Finished episode', (batch + 1) * batch_size,
            'Average rewards: ', avg_rewards)
        
        if realtime_render:
            env = None
            if env.gym:
                env = gym.make(env_type.env_name)
            else:
                env = env_type.make()
            state = env.reset()
            for t in range(500):
                env.render()
                policy = model([tf.convert_to_tensor([state])])
                action = np.random.choice(env_type.n_actions, p=policy.numpy()[0])
                state, _, done, _ = env.step(action)
                if done:
                    break
            env.close()
            
        # SAVE BEST MODEL -- if target reward is not reached
        if all(all(episode_reward >= er) for episode_reward in rewards for er in episode_reward_history):
            best_model = model
            # print("ADDED MODEL -- REWARDS: ", episode_reward)

        if avg_rewards >= reward_target:
            break

    return episode_reward_history, model, env, best_model

def train_deepq(reward_target: float, env_type: Environments.Environment, batch_size=16, n_episodes=1000, lr_in=0.001, lr_var=0.001, lr_out=0.1):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    observables = env_type.observables_func(ops)
    
    rlagent = DeepQLearning(learning_rate_in=lr_in, learning_rate_var=lr_var, learning_rate_out=lr_out)

    model = rlagent.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, False)
    model_target = rlagent.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, True)

    model_target.set_weights(model.get_weights())

    episode_reward_history = []
    step_count = 0
    
    env = []
    if env_type.gym:
        env = gym.make(env_type.env_name)
    else:
        env = env_type
    # env = gym.make(env_type.env_name)

    best_model = None

    for episode in range(n_episodes):
        episode_reward = 0
        state = env.reset()

        while True:
            # Interact with env
            interaction = rlagent.interact_env(state, model, rlagent.epsilon, env_type.n_actions, env)

            # Store interaction in the replay memory
            rlagent.replay_memory.append(interaction)

            state = interaction['next_state']
            episode_reward += interaction['reward']
            step_count += 1

            # Update model
            if step_count % rlagent.steps_per_update == 0:
                # Sample a batch of interactions and update Q_function
                training_batch = np.random.choice(rlagent.replay_memory, size=batch_size)
                rlagent.Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                np.asarray([x['action'] for x in training_batch]),
                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                np.asarray([x['next_state'] for x in training_batch]),
                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                model, rlagent.gamma, env_type.n_actions, model_target)

            # Update target model
            if step_count % rlagent.steps_per_target_update == 0:
                model_target.set_weights(model.get_weights())

            # Check if the episode is finished
            if interaction['done']:
                break

        # Decay epsilon
        # linear decay
        # rlagent.epsilon = min(rlagent.epsilon_max - 1.0 * (episode/n_episodes), rlagent.epsilon_min)
        # exponential decay
        rlagent.epsilon = max(rlagent.epsilon_min, rlagent.epsilon_max * (rlagent.epsilon_min / rlagent.epsilon_max)**(episode/n_episodes))
        # print("EPSILON: ", rlagent.epsilon)
        
        
        # SAVE BEST MODEL -- if target reward is not reached
        if all(episode_reward >= er for er in episode_reward_history):
            best_model = model
            # print("ADDED MODEL -- REWARDS: ", episode_reward)
            
        # ADD NEW EPISODE REWARD
        episode_reward_history.append(episode_reward)
        
        if (episode+1)%batch_size == 0:
            avg_rewards = np.mean(episode_reward_history[-batch_size:])
            print("Episode {}/{}, average last {} rewards {}".format(
                episode+1, n_episodes, batch_size, avg_rewards))
            if avg_rewards >= reward_target:
                break
    return episode_reward_history, model, env, rlagent

def export(history: list, env_type, model, train_method: TrainMethod, rlagent, dir="./images", episodes=0, note="", export_gif=True):
    if len(history) == 0:
        raise IndexError("Train a model first!")

    nr = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

    env = gym.make(env_type.env_name)
    state = env.reset()
    frames = []

    if train_method == TrainMethod.REINFORCE:
        for t in range(500):
            im = Image.fromarray(env.render(mode='rgb_array'))
            frames.append(im)
            policy = model([tf.convert_to_tensor([state/env_type.state_bounds])])
            action = np.random.choice(env_type.n_actions, p=policy.numpy()[0]) # only for two actions??
            state, _, done, _ = env.step(action)
            if done:
                break
    elif train_method == TrainMethod.DeepQLearning:
        for t in range(500):
            im = Image.fromarray(env.render(mode='rgb_array'))
            frames.append(im)
            # Use the model to predict the action probabilities
            action_probs = model([tf.convert_to_tensor([state])]) # /env_type.state_bounds
            action = int(tf.argmax(action_probs[0]).numpy())
            state, _, done, _ = env.step(action)
            if done:
                break
    env.close()
    if train_method == TrainMethod.REINFORCE:
        frames[0].save(f"{dir}/{nr}_gym_{env_type.env_name}_REINFORCE_batchSize=?_gamma=?_episodes={episodes}_{note}.gif",
                save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)
    elif train_method == TrainMethod.DeepQLearning:
        frames[1].save(f"{dir}/{nr}_gym_{env_type.env_name}_DeepQLearning_batchSize={rlagent.batch_size}_gamma={rlagent.gamma}_episodes={episodes}_learningrate_{[rlagent.learning_rate_in, rlagent.learning_rate_var, rlagent.learning_rate_out]}_{note}.gif",
                    save_all=True, append_images=frames[2:], optimize=False, duration=40, loop=0)