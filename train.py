from PolicyGradientRL import *
from DeepQRL import *
from PIL import Image
import os
import Environments
from enum import Enum

class TrainMethod(Enum):
    PolicyGradient = 0
    DeepQLearning = 1


episode_reward_history = []

def train(reward_target=500.0, realtime_render=False, batch_size=10, env_type=Environments.CartPole, method=TrainMethod.PolicyGradient):

    if method==TrainMethod.PolicyGradient:
        qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

        ops = [cirq.Z(q) for q in qubits]
        observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3

        model = generate_model_policy(qubits, env_type.n_layers, env_type.n_actions, 1.0, observables)

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

            avg_rewards = np.mean(episode_reward_history[-10:])

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
    elif method == TrainMethod.DeepQLearning:
        qubits = cirq.GridQubit.rect(1, env_type.n_qubits)

        ops = [cirq.Z(q) for q in qubits]
        observables = [ops[0]*ops[1], ops[2]*ops[3]] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1


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
            if (episode+1)%10 == 0:
                avg_rewards = np.mean(episode_reward_history[-10:])
                print("Episode {}/{}, average last 10 rewards {}".format(
                    episode+1, n_episodes, avg_rewards))
                if avg_rewards >= 500.0:
                    break
        return episode_reward_history, model, env

    else:
        raise ValueError("Unrecognized training method! Check the TrainMethod enum for valid methods.")

def export(history: list, env_type, model, dir="./images", note=""):
    if len(history) == 0:
        raise IndexError("Train a model first!")

    nr = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

    env = gym.make(env_type.env_name)
    state = env.reset()
    frames = []
    for t in range(500):
        im = Image.fromarray(env.render(mode='rgb_array'))
        frames.append(im)
        policy = model([tf.convert_to_tensor([state/env_type.state_bounds])])
        action = np.random.choice(env_type.n_actions, p=policy.numpy()[0])
        state, _, done, _ = env.step(action)
        if done:
            break
    env.close()
    frames[1].save(dir + '/' + str(nr) + '_gym_' + str(env_type.env_name) + str(note) + '.gif',
                save_all=True, append_images=frames[2:], optimize=False, duration=40, loop=0)