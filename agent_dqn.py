"""Tabular QL agent"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 300
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)

model = None
optimizer = None


def epsilon_greedy(state_vector, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (torch.FloatTensor): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    if np.random.rand() < epsilon:
        # Exploration: choisir aléatoirement une paire (action, objet)
        action_index = np.random.randint(0, NUM_ACTIONS - 1)
        object_index = np.random.randint(0, NUM_OBJECTS - 1)
    else:
        # Exploitation
        with torch.no_grad():
            q_values_action, q_values_object = model(state_vector)

            # Étend les Q-valeurs pour actions et objets pour former une matrice Q(a, b)
            q_matrix = q_values_action.unsqueeze(1) + q_values_object.unsqueeze(0)  # shape: (NUM_ACTIONS, NUM_OBJECTS)
            best_action, best_object = torch.argmax(q_matrix).item(), None
            action_index, object_index = np.unravel_index(best_action, (NUM_ACTIONS, NUM_OBJECTS))

    return action_index, object_index

class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q values for each (action, object) tuple given an input state vector
    """

    def __init__(self, state_dim, action_dim, object_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)
        self.state2object = nn.Linear(hidden_size, object_dim)

    def forward(self, x):
        state = F.relu(self.state_encoder(x))
        return self.state2action(state), self.state2object(state)


# pragma: coderesponse template
def deep_q_learning(current_state_vector, action_index, object_index, reward,
                    next_state_vector, terminal):
    """Updates the weights of the DQN for a given transition

    Args:
        current_state_vector (torch.FloatTensor): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (torch.FloatTensor): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    # 1. Compute the target value (y)
    with torch.no_grad():
        q_values_action_next, q_values_object_next = model(next_state_vector)
        max_q_action = q_values_action_next.max()
        max_q_object = q_values_object_next.max()
        maxq_next = 0.5 * (max_q_action + max_q_object)

        y = reward if terminal else reward + GAMMA * maxq_next

    # 2. Predict current Q(s, a, b)
    q_values_action_cur, q_values_object_cur = model(current_state_vector)
    q_action = q_values_action_cur[action_index]
    q_object = q_values_object_cur[object_index]
    q_current = 0.5 * (q_action + q_object)

    # 3. Define MSE loss: L = ½ (y - Q)^2
    loss = 0.5 * (y - q_current) ** 2

    # 4. Backpropagation and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# pragma: coderesponse end


def run_episode(for_training):
    """
        Runs one episode
        If for training, update Q function
        If for testing, computes and return cumulative discounted reward
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP
    epi_reward = 0.0 if not for_training else None
    discount = 1.0

    # Initial state
    current_room_desc, current_quest_desc, terminal = framework.newGame()

    while not terminal:
        # --- Vectorize current state ---
        current_state = current_room_desc + current_quest_desc
        current_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(current_state, dictionary)
        )

        # --- Choose action (a, b) ---
        action_index, object_index = epsilon_greedy(current_state_vector, epsilon)

        # --- Apply action ---
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index
        )

        # --- Vectorize next state ---
        next_state = next_room_desc + next_quest_desc
        next_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(next_state, dictionary)
        )

        # --- Train if needed ---
        if for_training:
            deep_q_learning(
                current_state_vector,
                action_index,
                object_index,
                reward,
                next_state_vector,
                terminal
            )

        # --- Accumulate reward for testing ---
        if not for_training:
            epi_reward += discount * reward
            discount *= GAMMA

        # --- Move to next state ---
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc

    if not for_training:
        return epi_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global model
    global optimizer
    model = DQN(state_dim, NUM_ACTIONS, NUM_OBJECTS)
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    state_texts = utils.load_data('game.tsv')
    dictionary = utils.bag_of_words(state_texts)
    state_dim = len(dictionary)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epsilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
