import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


# Add compatibility of Apple Silicon GPU, retaining compatibility with other systems
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    else:
        return torch.device("cpu")    # CPU fallback


# Builds user-to-item and item-to-user indices from the dataset
def build_index(dataset_name):
    """
    Builds indices for fast lookups of users' interacted items and items' associated users.
    Args:
        dataset_name: Name of the dataset file (without extension).
    Returns:
        u2i_index: List of items each user interacted with.
        i2u_index: List of users who interacted with each item.
    """
    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max() # Total number of users
    n_items = ui_mat[:, 1].max() # Total number of items

    # Initialize indices
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1]) # Add item to user's list
        i2u_index[ui_pair[1]].append(ui_pair[0]) # Add user to item's list

    return u2i_index, i2u_index

# sampler for batch generation
# Utility function to randomly sample a number not in a set
def random_neq(l, r, s):
    """
    Samples a random integer within [l, r) not in the set s.
    Args:
        l: Lower bound (inclusive).
        r: Upper bound (exclusive).
        s: Set of numbers to exclude.
    Returns:
        Random integer not in s.
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


# Sampling function for generating training batches
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    """
    Generates training batches with positive and negative samples.
    Args:
        user_train: Dictionary of users and their interaction sequences.
        usernum: Total number of users.
        itemnum: Total number of items.
        batch_size: Number of samples per batch.
        maxlen: Maximum sequence length.
        result_queue: Queue to store generated batches.
        SEED: Random seed for reproducibility.
    """
    def sample(uid):
        # Ensure the user has at least two interactions
        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32) # Interaction sequence
        pos = np.zeros([maxlen], dtype=np.int32) # Positive samples
        neg = np.zeros([maxlen], dtype=np.int32) # Negative samples
        nxt = user_train[uid][-1] # Last item in the sequence
        idx = maxlen - 1 # Index to fill sequence backwards

        ts = set(user_train[uid]) # Set of interacted items
        for i in reversed(user_train[uid][:-1]): # Iterate backward
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts) # Sample a negative item
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED) # Set random seed
    uids = np.arange(1, usernum+1, dtype=np.int32) # User IDs
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids) # Shuffle user IDs
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum])) # Sample user interactions
            counter += 1
        result_queue.put(zip(*one_batch)) # Add batch to the queue



# Wrapper class for parallelized sampling
class WarpSampler(object):
    """
    Handles batch sampling using multiprocessing for efficiency.
    """
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True # Daemon processes terminate with the main process
            self.processors[-1].start()

    def next_batch(self):
        """
        Fetches the next batch from the result queue.
        """
        return self.result_queue.get()

    def close(self):
        """
        Terminates all sampling processes.
        """
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
# Splits data into train, validation, and test sets
def data_partition(fname):
    """
    Partitions user interactions into training, validation, and test sets.
    Args:
        fname: Dataset filename (without extension).
    Returns:
        user_train: Training data.
        user_valid: Validation data.
        user_test: Test data.
        usernum: Number of users.
        itemnum: Number of items.
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    # Read dataset file
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    # Partition data for each user
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3: # Not enough interactions for validation/test
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else: # Split last two items for validation and test
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    """
    Evaluates the model on the test set using NDCG and Hit Rate.
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # Sample users for evaluation
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# Similar to evaluate(), but for validation data
def evaluate_valid(model, dataset, args):
    """
    Evaluates the model on the validation set using NDCG and Hit Rate.
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
