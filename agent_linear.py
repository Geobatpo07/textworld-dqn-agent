"""Linear QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False


GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 5
NUM_EPOCHS = 600
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.01  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def tuple2index(action_index, object_index):
    """Converts a tuple (a,b) to an index c"""
    return action_index * NUM_OBJECTS + object_index


def index2tuple(index):
    """Converts an index c to a tuple (a,b)"""
    return index // NUM_OBJECTS, index % NUM_OBJECTS


# pragma: coderesponse template name="linear_epsilon_greedy"
def epsilon_greedy(state_vector, theta, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (np.ndarray): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    """Returns an action selected by an epsilon-Greedy exploration policy."""
    if np.random.rand() < epsilon:
        # Exploration: choose random action-object pair
        action_index = np.random.randint(NUM_ACTIONS)
        object_index = np.random.randint(NUM_OBJECTS)
    else:
        # Exploitation: compute Q-values for all (action, object) pairs
        q_values = theta @ state_vector  # Shape: (NUM_ACTIONS * NUM_OBJECTS,)
        best_index = np.argmax(q_values)
        action_index, object_index = index2tuple(best_index)

    return action_index, object_index
# pragma: coderesponse end


# pragma: coderesponse template
def linear_q_learning(theta, current_state_vector, action_index, object_index,
                      reward, next_state_vector, terminal):
    """Update theta for a given transition

    Args:
        theta (np.ndarray): current weight matrix
        current_state_vector (np.ndarray): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (np.ndarray): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    # Convert (action_index, object_index) to flat index
    current_index = tuple2index(action_index, object_index)

    # Compute target y
    if terminal:
        y = reward
    else:
        q_next = theta @ next_state_vector  # shape: (NUM_ACTIONS * NUM_OBJECTS,)
        y = reward + GAMMA * np.max(q_next)

    # Current prediction
    q_current = theta[current_index] @ current_state_vector

    # Gradient descent step
    error = y - q_current
    theta[current_index] += ALPHA * error * current_state_vector
# pragma: coderesponse end


def run_episode(for_training):
    """ Runs one episode of the Home World game.
    If training, updates the Q-function using linear Q-learning.
    If testing, computes and returns cumulative discounted reward.

    Args:
        for_training (bool): True if for training, False if testing

    Returns:
        float or None: Cumulative discounted reward if testing, else None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP
    epi_reward = 0.0 if not for_training else None
    discount = 1.0

    # Initialize episode
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    while not terminal:
        # --- Encode current state ---
        current_state = current_room_desc + current_quest_desc
        current_state_vector = utils.extract_bow_feature_vector(current_state, dictionary)

        # --- Choose action ---
        action_index, object_index = epsilon_greedy(current_state_vector, theta, epsilon)

        # --- Take action ---
        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index)

        # --- Encode next state ---
        next_state = next_room_desc + next_quest_desc
        next_state_vector = utils.extract_bow_feature_vector(next_state, dictionary)

        # --- Update Q-function ---
        if for_training:
            linear_q_learning(theta, current_state_vector, action_index, object_index,
                              reward, next_state_vector, terminal)

        # --- Accumulate reward ---
        if not for_training:
            epi_reward += discount * reward
            discount *= GAMMA

        # --- Transition to next state ---
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
    global theta
    theta = np.zeros([action_dim, state_dim])

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
    action_dim = NUM_ACTIONS * NUM_OBJECTS

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test, axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epsilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
