# Q2-3
# # =========== Stochastic Gradient Descent : torch.optim.SGD and batch_size=1  ===============
# Accuracy of the network on the 10000 test images: 84 %
# Accuracy of Tshirt : 71 %
# Accuracy of Trouser : 96 %
# Accuracy of Pullover : 69 %
# Accuracy of Dress : 81 %
# Accuracy of  Coat : 84 %
# Accuracy of Sandal : 91 %
# Accuracy of Shirt : 69 %
# Accuracy of Sneaker : 95 %
# Accuracy of   Bag : 96 %
# Accuracy of Ankleboot : 92 %

# =========== Mini-Batch Gradient Descent  : torch.optim.ASGD and batch_size=64  ===============
# ====== Training Time 28.952475
# Accuracy of the network on the 10000 test images: 67 %
# Predicted:  Dress
# Accuracy of Tshirt : 55 %
# Accuracy of Trouser : 77 %
# Accuracy of Pullover : 64 %
# Accuracy of Dress : 87 %
# Accuracy of  Coat : 64 %
# Accuracy of Sandal : 11 %
# Accuracy of Shirt : 20 %
# Accuracy of Sneaker : 88 %
# Accuracy of   Bag : 87 %
# Accuracy of Ankleboot : 78 %

# --------------------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time

# --------------------------------------------------------------------------------------------
input_size = 28*28
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001
# --------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------
train_set = torchvision.datasets.FashionMNIST(root='./data_fashion', train=True, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data_fashion', train=False, download=True, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# # Find the right classes name. Save it as a tuple of size 10.
classes =("Tshirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankleboot")

# --------------------------------------------------------------------------------------------

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
plt.show()

print(' '.join('%5s' % classes[labels[j]] for j in range(1)))

# --------------------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()         # nn.LeakyReLU()        # nn.ELU()
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

torch.manual_seed(800)
net = Net(input_size, hidden_size, num_classes)
net.cuda()

##### -----  Q3 Implements Averaged Stochastic Gradient Descent.
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)        # stochastic gradient descent
optimizer = torch.optim.ASGD(net.parameters(), lr=learning_rate)
##### -----  Q3 Implements Averaged Stochastic Gradient Descent.

# --------------------------------------------------------------------------------------------
start = time.time()
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        images, labels = data
        images= images.view(-1, 28 * 28).cuda()
        images, labels = Variable(images), Variable(labels.cuda())
        optimizer.zero_grad()
        if batch_size > 1:
            for each in range(batch_size):
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        else:
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (i + 1) % 64 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.data[0]))

end = time.time()
print('====== Training Time %f' % (end-start))

# --------------------------------------------------------------------------------------------
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------

_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(1)))
# --------------------------------------------------------------------------------------------
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1, 28*28)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)
    for i in range(1):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')