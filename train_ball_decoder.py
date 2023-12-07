import gym
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import Adam
from functools import reduce
from collections import deque
from PIL import Image

from state_decoder import *

from Environments import AtariBreakout

# stack_size is the number of frames that will define the motion
stack_size = 4

def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu', input_shape=(3, 110, 84)))
    model.add(Convolution2D(64, (1, 1), activation='relu'))
    model.add(Convolution2D(64, (1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='softmax'))
    return model

def preprocess_frame(frame):
    # print(frame.shape)

    # Grayscale frame 
    gray = np.mean(frame, axis=2)

    # Crop the screen (remove the part below the player)
    cropped_frame = gray[8:-12,4:-12]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    normalized_frame = np.expand_dims(normalized_frame, axis=0)
    
    # Resize
    preprocessed_frame = np.resize(normalized_frame, [3,110,84])
    
    # preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    # print(preprocessed_frame.shape)
    return preprocessed_frame 

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((110,84), dtype=int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

# Initialize gym environment and the agent
env = gym.make(AtariBreakout.env_name)
height, width, channels = env.observation_space.shape
actions = env.action_space.n

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((110,84), dtype=int) for i in range(stack_size)], maxlen=4) 

# model = build_model(height, width, channels, actions)
# model.compile(optimizer=Adam(), loss='mse')


state = env.reset()
frames = []
for t in range(500):
    i = env.render(mode='rgb_array')
    im = Image.fromarray(i)
    frames.append(im)
    print(extract_state(i))
    env.step(1)

env.close()
frames[1].save('images/' + str(0) + '_gym_ALE_TEST' + '.gif',
            save_all=True, append_images=frames[2:], optimize=False, duration=40, loop=0)

# for episode in range(5):
#     state = env.reset()
#     env.metadata['render_fps'] = 30
#     done = False
#     while not done:
#         env.render(mode='rgb_array')
#         # Preprocess state in the same way as during training
#         processed_state = stack_frames(stacked_frames,state,)
#         # Use your trained model to select the action
#         action = np.argmax(model.predict(processed_state))
#         # Take action in the environment
#         next_state, reward, done, info = env.step(action)
#         state = next_state
#     print(f"Episode: {episode + 1}, Score: {info['score']}")
# env.close()