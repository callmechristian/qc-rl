from PolicyGradientRL import *
from DeepQRL import *
from PIL import Image
import os
import Environments
from enum import Enum
from scipy.special import softmax
import operator

class TrainMethod(Enum):
    PolicyGradient = 0
    DeepQLearning = 1

episode_reward_history = []

def train(reward_target=500.0, realtime_render=False, batch_size=10, env_type=Environments.CartPole, method=TrainMethod.PolicyGradient, n_episodes=1000):        

    if method==TrainMethod.PolicyGradient:
        if env_type == Environments.AtariBreakout:
            raise NotImplementedError("Policy Gradient not implemented for atari.")
            return train_policy_gradient_atari(reward_target, realtime_render, batch_size, env_type, n_episodes=n_episodes)
        else:
            return train_policy_gradient(reward_target, realtime_render, batch_size, env_type, n_episodes=n_episodes)
    elif method == TrainMethod.DeepQLearning:
        if env_type == Environments.AtariBreakout:
            return train_deepq_atari(reward_target, env_type, batch_size=batch_size, n_episodes=n_episodes)
            # raise NotImplementedError("Deep Q-Learning not implemented for atari.")
        else:
            return train_deepq(reward_target, env_type, batch_size=batch_size, n_episodes=n_episodes)
    else:
        raise ValueError("Unrecognized training method! Check the TrainMethod enum for valid methods.")

def train_policy_gradient(reward_target: float, realtime_render: bool, batch_size: int, env_type: Environments.Environment, n_episodes=1000):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3


    model = generate_model_policy(qubits, env_type.n_layers, env_type.n_actions, 1.0, observables)
    
    env = None

    episode_reward_history = []
    # Start training the agent
    for batch in range(n_episodes // batch_size):
        # Gather episodes
        episodes = gather_episodes(env_type.state_bounds, env_type.n_actions, model, batch_size, env_type.env_name)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([compute_returns(ep_rwds, gamma) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, model, batch_size)

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))

        avg_rewards = np.mean(episode_reward_history[-batch_size:])

        print('Finished episode', (batch + 1) * batch_size,
            'Average rewards: ', avg_rewards)
        
        if realtime_render:
            env = gym.make(env_type.env_name)
            state = env.reset()
            for t in range(500):
                env.render()
                policy = model([tf.convert_to_tensor([state/env_type.state_bounds])])
                action = np.random.choice(env_type.n_actions, p=policy.numpy()[0])
                state, _, done, _ = env.step(action)
                if done:
                    break
            env.close()

        if avg_rewards >= reward_target:
            break

    return episode_reward_history, model, env

def train_policy_gradient_atari(reward_target: float, realtime_render: bool, batch_size: int, env_type: Environments.Environment, n_episodes=1000):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3

    model = generate_model_policy(qubits, env_type.n_layers, env_type.n_actions, 1.0, observables)

    env = None

    episode_reward_history = []
    # Start training the agent
    for batch in range(n_episodes // batch_size):
        # Gather episodes
        episodes = gather_episodes(env_type.state_bounds, env_type.n_actions, model, batch_size, env_type.env_name, atari=True)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([compute_returns(ep_rwds, gamma) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, model, batch_size)

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
                    policy = model([tf.convert_to_tensor([extract_state(state)])])
                    action = np.random.choice(env_type.n_actions, p=policy.numpy()[0])
                    state, _, done, _ = env.step(action)
                    if done:
                        break
            finally:
                # print("Continuing without closing the window yet")
                if 'env' in locals():
                    env.close()

        # print('Continuing...')

        if avg_rewards >= reward_target:
            break

    return episode_reward_history, model, env

def train_deepq(reward_target: float, env_type: Environments.Environment, batch_size=16, n_episodes=1000):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    # observables = [ops[0]*ops[1], ops[2]*ops[3]] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    n_ops = len(ops) // env_type.n_actions  # number of ops to be multiplied together for each action
    # Generate observables for each action by chunks of ops
    observables = [reduce(operator.mul, chunk) for chunk in chunks(ops, n_ops)]
    print(len(observables))


    model = DeepQRL.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, False)
    model_target = DeepQRL.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, True)

    model_target.set_weights(model.get_weights())

    episode_reward_history = []
    step_count = 0
    env = gym.make(env_type.env_name)

    for episode in range(n_episodes):
        episode_reward = 0
        state = env.reset()

        while True:
            # Interact with env
            interaction = DeepQRL.interact_env(state, model, DeepQRL.epsilon, env_type.n_actions, env)

            # Store interaction in the replay memory
            DeepQRL.replay_memory.append(interaction)

            state = interaction['next_state']
            episode_reward += interaction['reward']
            step_count += 1

            # Update model
            if step_count % DeepQRL.steps_per_update == 0:
                # Sample a batch of interactions and update Q_function
                training_batch = np.random.choice(DeepQRL.replay_memory, size=batch_size)
                DeepQRL.Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                np.asarray([x['action'] for x in training_batch]),
                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                np.asarray([x['next_state'] for x in training_batch]),
                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                model, DeepQRL.gamma, env_type.n_actions, model_target)

            # Update target model
            if step_count % DeepQRL.steps_per_target_update == 0:
                model_target.set_weights(model.get_weights())

            # Check if the episode is finished
            if interaction['done']:
                break

        # Decay epsilon
        DeepQRL.epsilon = max(DeepQRL.epsilon * DeepQRL.decay_epsilon, DeepQRL.epsilon_min)
        episode_reward_history.append(episode_reward)
        if (episode+1)%batch_size == 0:
            avg_rewards = np.mean(episode_reward_history[-batch_size:])
            print("Episode {}/{}, average last {} rewards {}".format(
                episode+1, n_episodes, batch_size, avg_rewards))
            if avg_rewards >= reward_target:
                break
    return episode_reward_history, model, env

def train_deepq_atari(reward_target: float, env_type: Environments.Environment, batch_size=16, n_episodes=1000):
    qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1


    model = DeepQRL.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, False)
    model_target = DeepQRL.generate_model_Qlearning(qubits, env_type.n_layers, env_type.n_actions, observables, True)

    model_target.set_weights(model.get_weights())

    episode_reward_history = []
    step_count = 0
    env = gym.make(env_type.env_name)

    for episode in range(n_episodes):
        episode_reward = 0
        state = extract_state(env.reset())

        while True:
            # Interact with env
            interaction = DeepQRL.interact_env_atari(state, model, DeepQRL.epsilon, env_type.n_actions, env)

            # Store interaction in the replay memory
            DeepQRL.replay_memory.append(interaction)

            state = interaction['next_state']
            episode_reward += interaction['reward']
            step_count += 1

            # Update model
            if step_count % DeepQRL.steps_per_update == 0:
                # Sample a batch of interactions and update Q_function
                training_batch = np.random.choice(DeepQRL.replay_memory, size=batch_size)
                DeepQRL.Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                np.asarray([x['action'] for x in training_batch]),
                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                np.asarray([x['next_state'] for x in training_batch]),
                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                model, DeepQRL.gamma, env_type.n_actions, model_target)

            # Update target model
            if step_count % DeepQRL.steps_per_target_update == 0:
                model_target.set_weights(model.get_weights())

            # Check if the episode is finished
            if interaction['done']:
                break

        # Decay epsilon
        DeepQRL.epsilon = max(DeepQRL.epsilon * DeepQRL.decay_epsilon, DeepQRL.epsilon_min)
        episode_reward_history.append(episode_reward)
        if (episode+1)%batch_size == 0:
            avg_rewards = np.mean(episode_reward_history[-batch_size:])
            print("Episode {}/{}, average last {} rewards {}".format(
                episode+1, n_episodes, batch_size, avg_rewards))
            if avg_rewards >= reward_target:
                break
    return episode_reward_history, model, env

def export(history: list, env_type, model, train_method: TrainMethod, dir="./images", note=""):
    if len(history) == 0:
        raise IndexError("Train a model first!")

    nr = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

    env = gym.make(env_type.env_name)
    state = env.reset()
    frames = []

    if train_method == TrainMethod.PolicyGradient:
        for t in range(500):
            im = Image.fromarray(env.render(mode='rgb_array'))
            frames.append(im)
            policy = model([tf.convert_to_tensor([state/env_type.state_bounds])])
            action = np.random.choice(env_type.n_actions, p=policy.numpy()[0])
            state, _, done, _ = env.step(action)
            if done:
                break
    elif train_method == TrainMethod.DeepQLearning:
        for t in range(500):
            im = Image.fromarray(env.render(mode='rgb_array'))
            frames.append(im)
            # Use the model to predict the action probabilities
            action_probs = model([tf.convert_to_tensor([state/env_type.state_bounds])])
            action = int(tf.argmax(action_probs[0]).numpy())
            state, _, done, _ = env.step(action)
            if done:
                break
    env.close()
    frames[1].save(dir + '/' + str(nr) + '_gym_' + str(env_type.env_name) + str(note) + '.gif',
                save_all=True, append_images=frames[2:], optimize=False, duration=40, loop=0)