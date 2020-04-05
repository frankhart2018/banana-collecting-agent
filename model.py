# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Q-Network class
class QNetwork(nn.Module):

    """
    Actor (Policy) Model
    """

    def __init__(self, state_size, action_size, seed):

        """
        Initialize parameters and build model

        Params
        ======
            state_size (int):  dimension of each state
            action_size (int): dimension of each action
            seed (int):        random seed
        """

        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Define the layers of neural network
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):

        """
        Build a network that maps state -> action values
        """

        # Forwar pass state through various layers to get action
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
