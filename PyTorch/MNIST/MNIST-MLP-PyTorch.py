# --------- MLP MNIST

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

# number of workers
num_worker=0
# Batch size
batch_size=10
# percentage of training to use validation
valid_size=0.2
# input size
num_input = 28 * 28
# number of network
num_network=512
# number of target
num_target = 10
# Learning rate
learning_rate = 0.001
# Number of epochs
num_epochs = 50


# Get training and test data
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

num_train = len(train_data)
num_train_list = list(range(num_train))
np.random.shuffle(num_train_list)
split = int(np.floor(valid_size*num_train))
train_idx, valid_idx = num_train_list[split:], num_train_list[:split]


# Get samples from train and test data
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_worker)
valid_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_worker)
test_load = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_worker)

import matplotlib.pyplot as plt

trainiter = iter(train_load)
images, labels = trainiter.next()
images = images.numpy()

# 20 Sample show plot
fig = plt.figure(figsize=(50,8))
for i in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, i+1)
    ax.imshow(np.squeeze(images[i]), cmap='gray')
    ax.set_title(str(labels[i].item()))

plt.show()

# Create Model
class Net(nn.Module):
    def __init__(self, num_input, num_network, num_target):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_input, num_network)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_network, num_target)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Init model
net = Net(num_input, num_network, num_target)
print(net)
# Define loss and optimizer function
creterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# ---- Train Network
valid_loss_min = np.Inf
for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.00
    #
    for i, (images, labels) in enumerate(train_load):
        # Convert torch tensor to Variable
        images = images.view(-1, 28 * 28)

        # clear the gradients
        optimizer.zero_grad()
        # get predicted output
        output = net(images)
        # loss
        loss = creterion(output, labels)
        # backward loss: compute graidents
        loss.backward()
        # perform optimizer : update parameters
        optimizer.step()
        # update training loss
        train_loss += loss.item()*images.size(0)
    #
    net.eval()
    for i, (images, labels) in enumerate(valid_load):
        # Convert torch tensor to Variable
        images = images.view(-1, 28 * 28)

        # get predicted output
        output = net(images)
        # loss
        loss = creterion(output, labels)
        # update validation loss
        valid_loss += loss.item()*images.size(0)

    #
    train_loss = train_loss/len(train_load.dataset)
    valid_loss = valid_loss/len(valid_load.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1,
        train_loss,
        valid_loss
        ))

    if valid_loss < valid_loss_min:
        print('Validation loss changed: ', valid_loss)
        valid_loss_min = valid_loss
        torch.save(net.state_dict(), 'model.pt')

# --- Load model with lowest validation loss
net.load_state_dict(torch.load('model.pt'))

# --- Test Model
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for i, (images, labels) in enumerate(valid_load):
    # Convert torch tensor to Variable
    images = images.view(-1, 28 * 28)
    # compute predicted output
    output = net(images)
    # loss
    loss = creterion(output, labels)
    # update test loss
    test_loss += loss.item()*images.size(0)
    # get probability
    _, pred = torch.max(output, 1)
    # ????
    correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
    # calculate test accuracy
    for i in range(batch_size):
        label=labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] +=1

test_loss = test_loss/len(test_load.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

# for i in range(10):
#     if class_total[i] > 0:
#         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#             str(i), 100 * class_correct[i] / class_total[i],
#             np.sum(class_correct[i]), np.sum(class_total[i])))
#     else:
#         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy : %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

