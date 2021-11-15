import pandas as pd
from data_loader import *
from time import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# HYPERPARAMETERS
MAX_SENTENCE_LENGTH = 850
LEARNING_RATE = 0.001
DECAY = 5e-5
HIDDEN_1 = 264
HIDDEN_2 = 264
HIDDEN_3 = 264
BATCH_SIZE = 50
EPOCHS = 1000
TEST_SPLIT = 10000 # How much of our data should we test on

# Read IMDB csv
df = pd.read_csv('data/imdb.csv')
df['y_output'] = (df['sentiment'] == 'positive') * 1

# Create Word Index Dictionary
df_word_index = pd.read_csv('cleaned-word-dict.csv', index_col=0)
index_word = df_word_index.to_dict('dict')['WORD']  # gets rid of the nested dict and makes sure that we only take the word column (the key is an int, value is the desired word)
word_index = flip_dict(index_word)

# Prepare Data
print('Preparing Data...')

df['review'] = df['review'].apply(lambda string: string.lower())  # makes all of the strings in 'review' to lower case

inputs = torch.Tensor(create_input_matrix(list(df['review']), word_index, MAX_SENTENCE_LENGTH))
lables = list(df['y_output'])

X_test = torch.tensor(inputs[:TEST_SPLIT])
X_test_norm = (X_test - (MAX_SENTENCE_LENGTH / 2)) / (MAX_SENTENCE_LENGTH / 2)
y_test = torch.tensor(lables[:TEST_SPLIT])

X_train = torch.tensor(inputs[TEST_SPLIT:])
X_train_norm = (X_train - (MAX_SENTENCE_LENGTH / 2)) / (MAX_SENTENCE_LENGTH / 2)
y_train = torch.tensor(lables[TEST_SPLIT:])

# Model Creation

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MAX_SENTENCE_LENGTH, HIDDEN_1)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.fc3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.fc4 = nn.Linear(HIDDEN_3, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

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
    for batch in range(len(X_train) // BATCH_SIZE):
        X_batch = X_train[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
        y_batch = y_train[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
        net.zero_grad()
        output = net(X_batch.to(device))
        loss = F.nll_loss(output, y_batch.to(device))
        loss.backward()
        optimizer.step()
    if not epoch % 10:
        print(f'Epoch: {epoch}, Loss: {loss:.5}')

# Testing
print('Testing...')
correct = 0
total = 0
with torch.no_grad():
    output = net(X_test.to(device))
    for idx, i in enumerate(output):
        if torch.argmax(i).to(device) == y_test[idx].to(device):
            correct += 1
        total += 1
    print(f'Accuracy: {(correct / total):.3}')

# Testing on trainset - Did it meorize?
correct = 0
total = 0
with torch.no_grad():

    output = net(X_train.to(device))
    for idx, i in enumerate(output):
        if torch.argmax(i).to(device) == y_train[idx].to(device):
            correct += 1
        total += 1
    print(f'Accuracy in trainset: {(correct / total):.3}')
