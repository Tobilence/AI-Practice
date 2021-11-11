import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import transforms, datasets
from time import time

train = datasets.MNIST("", train=True, download=True, transform= transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform= transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)



# Net Declaration
net = Net(784)
optimizer = optim.Adam(net.parameters(), lr=0.001)
EPOCHS = 3

t1 = time()
print('Starting training...')

# Training
for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of featuresets and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 784))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

t2 = time()
print(f'Finished Training in {(t2-t1): .3} seconds')

correct = 0
total = 0

# Testing
print('Testing model...')
with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
    print(f'Accuracy: {round(correct / total, 3)}')

t3 = time()
print(f'Calculated accuracy in {t3-t2: .3} seconds')

# Predict
print(torch.argmax(net(X[3].view(-1, 784))[0]))








