from PolicyGradientRL import *
from PIL import Image
import os

model = generate_model_policy(qubits, n_layers, n_actions, 1.0, observables)
episode_reward_history = []

def train(reward_target=500.0, realtime_render=False, batch_size=10, env_name="CartPole-v1"):
    episode_reward_history = []
    # Start training the agent
    for batch in range(n_episodes // batch_size):
        # Gather episodes
        episodes = gather_episodes(state_bounds, n_actions, model, batch_size, env_name)

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

        if avg_rewards >= reward_target:
            break

        if realtime_render:
            env = gym.make(env_name)
            state = env.reset()
            for t in range(500):
                env.render()
                policy = model([tf.convert_to_tensor([state/state_bounds])])
                action = np.random.choice(n_actions, p=policy.numpy()[0])
                state, _, done, _ = env.step(action)
                if done:
                    break
            env.close()

    return episode_reward_history

def export(history: list, dir="./images", note="", env_name="CartPole-v1"):
    if len(history) == 0:
        raise IndexError("Train a model first!")

    nr = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])

    env = gym.make(env_name)
    state = env.reset()
    frames = []
    for t in range(500):
        im = Image.fromarray(env.render(mode='rgb_array'))
        frames.append(im)
        policy = model([tf.convert_to_tensor([state/state_bounds])])
        action = np.random.choice(n_actions, p=policy.numpy()[0])
        state, _, done, _ = env.step(action)
        if done:
            break
    env.close()
    frames[1].save(dir + '/' + str(nr) + '_gym_' + str(env_name) + str(note) + '.gif',
                save_all=True, append_images=frames[2:], optimize=False, duration=40, loop=0)
