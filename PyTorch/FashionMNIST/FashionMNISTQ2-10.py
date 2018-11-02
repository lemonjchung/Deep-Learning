# Q2-10
# Confusion matrix, without normalization
# [[973   0   0   0   0   3   1  23   0   0]
#  [  0 956   4   5  11   3  10   4   1   6]
#  [  0   6 758  34 122   0  80   0   0   0]
#  [  0   6  30 880  13   0  27   0   6  38]
#  [  0   4  86   8 794   0  92   0   1  15]
#  [ 20   2   0   1   0 952   0  24   0   1]
#  [  0  11  59  26  89   0 626   0   1 188]
#  [ 64   0   0   0   0  27   0 909   0   0]
#  [  0   1   4  23   3   0   1   0 963   5]
#  [  0  10   3  19  19   1  74   0   0 874]]
# Normalized confusion matrix
# [[0.973 0.    0.    0.    0.    0.003 0.001 0.023 0.    0.   ]
#  [0.    0.956 0.004 0.005 0.011 0.003 0.01  0.004 0.001 0.006]
#  [0.    0.006 0.758 0.034 0.122 0.    0.08  0.    0.    0.   ]
#  [0.    0.006 0.03  0.88  0.013 0.    0.027 0.    0.006 0.038]
#  [0.    0.004 0.086 0.008 0.794 0.    0.092 0.    0.001 0.015]
#  [0.02  0.002 0.    0.001 0.    0.952 0.    0.024 0.    0.001]
#  [0.    0.011 0.059 0.026 0.089 0.    0.626 0.    0.001 0.188]
#  [0.064 0.    0.    0.    0.    0.027 0.    0.909 0.    0.   ]
#  [0.    0.001 0.004 0.023 0.003 0.    0.001 0.    0.963 0.005]
#  [0.    0.01  0.003 0.019 0.019 0.001 0.074 0.    0.    0.874]]


# --------------------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

# --------------------------------------------------------------------------------------------
net = Net(input_size, hidden_size, num_classes)
net.cuda()

# --------------------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# --------------------------------------------------------------------------------------------
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
# There is bug here find it and fix it
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
y_predict = []
y_test = []

for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1, 28*28)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)

    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)

    '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    for i in range(batch_size):
        y_test.append(classes[labels[i]])
        y_predict.append(classes[predicted.cpu()[i]])
    '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predict)
#np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()