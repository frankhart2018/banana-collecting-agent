# Import libraries
import torch
from unityagents import UnityEnvironment
import numpy as np
import random

from model import QNetwork

# Load the environment
env = UnityEnvironment(file_name="Banana.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Load trained Q-Network
from model import QNetwork

qnetwork = QNetwork(state_size=37, action_size=4, seed=0)
qnetwork.load_state_dict(torch.load('checkpoint.pth'))

# Testing
def test(state, eps):

    """
    Test the agent

    Params
    ======
        state (numpy.ndarray): Current state of the agent in the environment
        eps (float):           Epsilon for epsilon-greedy environment
    """

    global qnetwork

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        qnetwork = qnetwork.cuda()
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    qnetwork.eval()
    with torch.no_grad():
        action_values = qnetwork(state)
    if random.random() > eps:
        return np.argmax(action_values.cpu().data.numpy())
    else:
        return random.choice(np.arange(4))

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = test(state, 0.01)                           # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
env.close()

print("Score: {}".format(score))
