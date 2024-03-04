from algorithms import REINFORCE, DeepQLearning
import data.Environments as Environments
import gym, cirq
from functools import reduce
from utils import state_decoder
import numpy as np
import tensorflow as tf

def train_deepq_atari(reward_target: float, env_type: Environments.Environment, batch_size=16, n_episodes=1000):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    observables = env_type.observables_func(ops)


    model = DeepQLearning.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, False)
    model_target = DeepQLearning.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, True)

    model_target.set_weights(model.get_weights())

    episode_reward_history = []
    step_count = 0
    env = gym.make(env_type.env_name)

    for episode in range(n_episodes):
        episode_reward = 0
        state = state_decoder.extract_state(env.reset())

        while True:
            # Interact with env
            interaction = DeepQLearning.interact_env_atari(state, model, DeepQLearning.epsilon, env_type.n_actions, env)

            # Store interaction in the replay memory
            DeepQLearning.replay_memory.append(interaction)

            state = interaction['next_state']
            episode_reward += interaction['reward']
            step_count += 1

            # Update model
            if step_count % DeepQLearning.steps_per_update == 0:
                # Sample a batch of interactions and update Q_function
                training_batch = np.random.choice(DeepQLearning.replay_memory, size=batch_size)
                DeepQLearning.Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                np.asarray([x['action'] for x in training_batch]),
                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                np.asarray([x['next_state'] for x in training_batch]),
                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                model, DeepQLearning.gamma, env_type.n_actions, model_target)

            # Update target model
            if step_count % DeepQLearning.steps_per_target_update == 0:
                model_target.set_weights(model.get_weights())

            # Check if the episode is finished
            if interaction['done']:
                break

        # Decay epsilon
        DeepQLearning.epsilon = max(DeepQLearning.epsilon * DeepQLearning.decay_epsilon, DeepQLearning.epsilon_min)
        episode_reward_history.append(episode_reward)
        if (episode+1)%batch_size == 0:
            avg_rewards = np.mean(episode_reward_history[-batch_size:])
            print("Episode {}/{}, average last {} rewards {}".format(
                episode+1, n_episodes, batch_size, avg_rewards))
            if avg_rewards >= reward_target:
                break
    return episode_reward_history, model, env

def train_policy_gradient_atari(reward_target: float, realtime_render: bool, batch_size: int, env_type: Environments.Environment, n_episodes=1000):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops), -reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3

    model = REINFORCE.generate_model_policy(qubits, env_type.n_layers, env_type.n_actions, 1.0, observables)

    env = None

    episode_reward_history = []
    # Start training the agent
    for batch in range(n_episodes // batch_size):
        # Gather episodes
        episodes = REINFORCE.gather_episodes(env_type.state_bounds, env_type.n_actions, model, batch_size, env_type.env_name, atari=True)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([REINFORCE.compute_returns(ep_rwds, REINFORCE.gamma) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        REINFORCE.reinforce_update(states, id_action_pairs, returns, model, batch_size)

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))

        avg_rewards = np.mean(episode_reward_history[-batch_size:])

        print('Finished episode', (batch + 1) * batch_size,
            'Average rewards: ', avg_rewards)
        
        if realtime_render:
            try:
                env = gym.make(env_type.env_name, render_mode='human')
                env.metadata['render_fps'] = 60
                state = env.reset()

            
                for t in range(500):
                    policy = model([tf.convert_to_tensor([state_decoder.extract_state(state)])])
                    action = np.random.choice(env_type.n_actions, p=policy.numpy()[0])
                    state, _, done, _ = env.step(action)
                    if done:
                        break
            finally:
                if 'env' in locals():
                    env.close()

        if avg_rewards >= reward_target:
            break

    return episode_reward_history, model, env