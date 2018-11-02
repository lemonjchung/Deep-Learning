# ========== NO dropout ==================================================================
# https://pytorch.org/docs/stable/nn.html
# Improving neural networks by preventing co-adaptation of feature detectors : https://arxiv.org/abs/1207.0580
# We initially explored the effectiveness of dropout using MNIST, a widely used benchmark
# for machine learning algorithms. It contains 60,000 28x28 training images of individual hand
# written digits and 10,000 test images. Performance on the test set can be greatly improved by
# enhancing the training data with transformed images (3) or by wiring knowledge about spatial
# transformations into a convolutional neural network (4) or by using generative pre-training to
# extract useful features from the training images without using the labels (5). Without using any
# of these tricks, the best published result for a standard feedforward neural network is 160 errors
# on the test set. This can be reduced to about 130 errors by using 50% dropout with separate L2
# constraints on the incoming weights of each hidden unit and further reduced to about 110 errors
# by also dropping out a random 20% of the pixels (see figure 1).


# •	Dropout(1)
# Accuracy of the network on the 10000 test images: 10 %
# Predicted:  Sneaker Sneaker Sneaker Sneaker
# Accuracy of Tshirt :  0 %
# Accuracy of Trouser :  0 %
# Accuracy of Pullover :  0 %
# Accuracy of Dress :  0 %
# Accuracy of  Coat :  0 %
# Accuracy of Sandal :  0 %
# Accuracy of Shirt :  0 %
# Accuracy of Sneaker : 100 %
# Accuracy of   Bag :  0 %
# Accuracy of Ankleboot :  0 %
#
# •	Dropout(0.1)
# Accuracy of the network on the 10000 test images: 85 %
# Predicted:  Shirt Dress Shirt   Bag
# Accuracy of Tshirt : 74 %
# Accuracy of Trouser : 98 %
# Accuracy of Pullover : 71 %
# Accuracy of Dress : 94 %
# Accuracy of  Coat : 83 %
# Accuracy of Sandal : 89 %
# Accuracy of Shirt : 64 %
# Accuracy of Sneaker : 90 %
# Accuracy of   Bag : 98 %
# Accuracy of Ankleboot : 100 %
#
#
# •	Dropout(0.9)
# Accuracy of the network on the 10000 test images: 54 %
# Predicted:  Tshirt Dress Dress   Bag
# Accuracy of Tshirt : 51 %
# Accuracy of Trouser : 84 %
# Accuracy of Pullover : 30 %
# Accuracy of Dress : 69 %
# Accuracy of  Coat : 62 %
# Accuracy of Sandal : 61 %
# Accuracy of Shirt :  9 %
# Accuracy of Sneaker : 67 %
# Accuracy of   Bag : 68 %
# Accuracy of Ankleboot : 75 %



# ====== Training Time 27.936551
# Accuracy of the network on the 10000 test images: 86 %
# Predicted:  Shirt Dress Tshirt   Bag
# Accuracy of Tshirt : 80 %
# Accuracy of Trouser : 98 %
# Accuracy of Pullover : 79 %
# Accuracy of Dress : 97 %
# Accuracy of  Coat : 86 %
# Accuracy of Sandal : 92 %
# Accuracy of Shirt : 54 %
# Accuracy of Sneaker : 83 %
# Accuracy of   Bag : 98 %
# Accuracy of Ankleboot : 96 %

# ====== Training Time 27.232856
# Accuracy of the network on the 10000 test images: 86 %
# Predicted:  Shirt Dress Tshirt   Bag
# Accuracy of Tshirt : 94 %
# Accuracy of Trouser : 98 %
# Accuracy of Pullover : 79 %
# Accuracy of Dress : 91 %
# Accuracy of  Coat : 79 %
# Accuracy of Sandal : 87 %
# Accuracy of Shirt : 64 %
# Accuracy of Sneaker : 88 %
# Accuracy of   Bag : 98 %
# Accuracy of Ankleboot : 100 %


# ========== nn.Dropout(0.2) ==================================================================
# https://www.programcreek.com/python/example/107689/torch.nn.Dropout
# ====== Training Time 29.845960
# Accuracy of the network on the 10000 test images: 85 %
# Predicted:  Shirt Dress Shirt   Bag
# Accuracy of Tshirt : 80 %
# Accuracy of Trouser : 98 %
# Accuracy of Pullover : 84 %
# Accuracy of Dress : 97 %
# Accuracy of  Coat : 74 %
# Accuracy of Sandal : 89 %
# Accuracy of Shirt : 45 %
# Accuracy of Sneaker : 86 %
# Accuracy of   Bag : 98 %
# Accuracy of Ankleboot : 100 %

# ====== Training Time 28.652802
# Accuracy of the network on the 10000 test images: 86 %
# Predicted:  Shirt Dress Shirt   Bag
# Accuracy of Tshirt : 80 %
# Accuracy of Trouser : 98 %
# Accuracy of Pullover : 76 %
# Accuracy of Dress : 97 %
# Accuracy of  Coat : 83 %
# Accuracy of Sandal : 89 %
# Accuracy of Shirt : 61 %
# Accuracy of Sneaker : 83 %
# Accuracy of   Bag : 98 %
# Accuracy of Ankleboot : 93 %

# ========== nn.Dropout(0.5) ==================================================================
# ====== Training Time 28.062566
# Accuracy of the network on the 10000 test images: 83 %
# Predicted:  Shirt Dress Shirt   Bag
# Accuracy of Tshirt : 74 %
# Accuracy of Trouser : 96 %
# Accuracy of Pullover : 82 %
# Accuracy of Dress : 97 %
# Accuracy of  Coat : 69 %
# Accuracy of Sandal : 89 %
# Accuracy of Shirt : 51 %
# Accuracy of Sneaker : 90 %
# Accuracy of   Bag : 94 %
# Accuracy of Ankleboot : 96 %


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
# Choose the right values for x.
input_size = 28*28
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 100
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

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# --------------------------------------------------------------------------------------------
# Choose the right argument for xx
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.dropdout = nn.Dropout(0.9)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()         # nn.LeakyReLU()        # nn.ELU()
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropdout(out)
        out = self.fc2(out)

        return out

# --------------------------------------------------------------------------------------------
net = Net(input_size, hidden_size, num_classes)
net.cuda()

# --------------------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
start = time.time()
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        images, labels = data
        images= images.view(-1, 28 * 28).cuda()
        images, labels = Variable(images), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
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
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
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
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')