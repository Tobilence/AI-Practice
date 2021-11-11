import pandas as pd
from data_loader import *
from time import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# HYPERPARAMETERS
MAX_SENTENCE_LENGTH = 650
LEARNING_RATE = 0.001
DECAY = 5e-5
HIDDEN_1 = 64
HIDDEN_2 = 64
HIDDEN_3 = 64
BATCH_SIZE = 50
EPOCHS = 200
TEST_SPLIT = 10000 # How much of our data should we test on

# Read IMDB csv
df = pd.read_csv('data/imdb.csv')
df['y_output'] = (df['sentiment'] == 'positive') * 1

# Create Word Index Dictionary
df_word_index = pd.read_csv('progress_dict_20000.csv', index_col=0)
index_word = df_word_index.to_dict('dict')['0']
word_index = flip_dict(index_word)

# Prepare Data
print('Preparing Data...')
inputs = torch.Tensor(create_input_matrix(list(df['review']), word_index, 650))
lables = list(df['y_output'])

X_test = torch.tensor(inputs[:TEST_SPLIT])
y_test = torch.tensor(lables[:TEST_SPLIT])
X_train = torch.tensor(inputs[TEST_SPLIT:])
y_train = torch.tensor(lables[TEST_SPLIT:])

print(len(X_train), len(y_train), len(X_test), len(y_test))

# Model Creation

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MAX_SENTENCE_LENGTH, HIDDEN_1)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.fc3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.fc4 = nn.Linear(HIDDEN_3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# - Check if GPU is Available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Pytorch is running on GPU")
else:
    device = torch.device("cpu")
    print("Pytorch is running on CPU")

# - Create Model
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Training
print('Training...')
for epoch in range(EPOCHS):
    t1 = time()
    # BATCHING - TODO! for batch_nr in range(MAX_SENTENCE_LENGTH // BATCH_SIZE):
    net.zero_grad()
    output = net.forward(X_train)
    loss = F.nll_loss(output, y_train)
    loss.backward()
    optimizer.step()
    t2 = time()
    print(f'Loss: {loss:.5}, this epoch took: {t2 - t1} seconds')

# Testing
print('Testing...')
correct = 0
total = 0
with torch.no_grad():

    output = net(X_test)
    for idx, i in enumerate(output):
        if torch.argmax(i).to(device) == y_test[idx]:
            correct += 1
        total += 1
    print(f'Accuracy: {(correct / total):.3}')

# Testing on trainset - Did it meorize?
print('Testing...')
correct = 0
total = 0
with torch.no_grad():

    output = net(X_train)
    for idx, i in enumerate(output):
        if torch.argmax(i).to(device) == y_train[idx]:
            correct += 1
        total += 1
    print(f'Accuracy in trainset: {(correct / total):.3}')
