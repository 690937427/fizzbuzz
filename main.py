import torch
import torch.nn.functional as F
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", 'fizzbuzz'][prediction]


def helper(i):
    return fizz_buzz_decode(i, fizz_buzz_encode(i))


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])


NUM_DIGTIS = 10
BATCH_SIZE = 32
trX = torch.tensor(np.array([binary_encode(i, NUM_DIGTIS) for i in range(101, 2 ** NUM_DIGTIS)]), dtype=torch.float32)
trY = torch.tensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGTIS)])
net = Net(n_feature=10, n_hidden=100, n_output=4)  # define the network
print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()  # this is for regression mean squared loss

for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]
        prediction = net(batchX)  # input x and predict based on x
        loss = loss_func(prediction, batchY)  # must be (1. nn output, 2. target)
        print('Epoch:', epoch, (loss.item()))
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
testX = torch.tensor(np.array([binary_encode(i, NUM_DIGTIS) for i in range(1, 101)]), dtype=torch.float32)
with torch.no_grad():
    testY = net(testX)
predict = zip(range(1, 101), testY.max(1)[1].data.tolist())
print([fizz_buzz_decode(i, x) for i, x in list(predict)])
