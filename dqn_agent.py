# Import libraries
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork

# Set hyperparameters
BUFFER_SIZE = int(1e5)  # Maximum size of the replay buffer
BATCH_SIZE = 64         # Batch size for sampling from replay buffer
GAMMA = 0.99            # Discount factor for calculating return
TAU = 1e-3              # Hyperparameter for soft update of target parameters
LR = 5e-4               # Learning rate for the neural networks
UPDATE_EVERY = 4        # Number of time steps after which soft update is performed

# Set the device on which to run the neural network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the agent class
class Agent():

    """
    The banana collecting agent
    """

    def __init__(self, state_size, action_size, seed):

        """
        Initialize an Agent object

        Params
        ======
            state_size (int):  dimension of each state
            action_size (int): dimension of each action
            seed (int):        random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Define the Q-Networks and move to available device (CPU/GPU)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        # Define the optimizer for the local Q-Network
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR)

        # Create Replay buffer object
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for checking when to update parameters of target Q-Network)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        """
        Take a step in Q-Network

        Params
        ======
            state (numpy.ndarray): current state of the agent
            action (int):          action taken in the current state
            reward (int):          reward received after taking an action in the current state
            next_state (int):      next state to which agent goes after performing an action in the current state
            done (bool):           check if agent reached goal state, this marks the end of an episode
        """

        # Save current experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn after every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # If there are enough samples in the replay buffer then sample randomly from it
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0):

        """
        Returns actions for given state as per current policy

        Params
        ======
            state (numpy.ndarray): current state of the agent
            eps (float):           epsilon, for epsilon-greedy action selection
        """

        # Convert state to torch tensor, add extra dimension, and put to available device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Set local Q-Network into evaluation mode for prediction
        self.qnetwork_local.eval()

        # Get output without requiring any gradients
        with torch.no_grad():
            # Get the action values from local Q-Network
            action_values = self.qnetwork_local(state)

        # Set local Q-Network into train mode for the next training step
        self.qnetwork_local.train()

        # Select action using epsilon-greedy policy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        """
        Update value parameters using given batch of experience tuples

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of state, action, reward, next state, done tensors
            gamma (float):                     discount factor
        """

        # Assign the tuple values into variables
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values from the target Q-Network
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get the expected Q values from local Q-Network
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Define the mean squared error loss between expected and target Q values
        loss = F.mse_loss(Q_expected, Q_targets)

        # Compute loss and backpropagate errors
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):

        """
        Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (torch model):  local model
            target_model (torch model): target model
            tau (float):                interpolation parameter for soft update
        """

        # Perform soft update
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Define the replay buffer class
class ReplayBuffer:

    """
    Fixed-size buffer to store experience tuples
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):

        """
        Initialize a ReplayBuffer object

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int):  size of each training batch
            seed (int):        random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):

        """
        Add a new experience to memory

        Params
        ======
            state (numpy.ndarray): current state of the agent
            action (int):          action taken in the current state
            reward (int):          reward received after taking an action in the current state
            next_state (int):      next state to which agent goes after performing an action in the current state
            done (bool):           check if agent reached goal state, this marks the end of an episode
        """

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):

        """
        Randomly sample a batch of experiences from memory
        """

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Return the current size of internal memory
        """

        return len(self.memory)
