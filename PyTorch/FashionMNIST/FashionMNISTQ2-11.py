# Q 2-11

# CODE
# # Q2-11 show 10 classes images with output
# def showImageAndOutput(classes, classesimages, correctlist, totallist):
#     plotcolumns = 5
#     plotrows = int(len(classes) / plotcolumns)
#
#     fig, axes = plt.subplots(nrows=plotrows, ncols=plotcolumns, figsize=(20, 30))
#     for ci in range(len(classes)):
#         subplot_row = ci // plotcolumns   # plot row
#         subplot_col = ci % plotcolumns    # plot column
#
#         axes[subplot_row, subplot_col].text(0, 32, classes[ci], fontsize=20)
#         axes[subplot_row, subplot_col].text(0, 36, 'Correct#/Total# : %d / %d' %(correctlist[ci], totallist[ci]), fontsize=15)
#         axes[subplot_row, subplot_col].axis('off')
#         img = classesimages[classes[ci]].cpu().numpy()
#         X = np.reshape(img, (28, 28))
#         axes[subplot_row, subplot_col].imshow(X, cmap='gray')
#     plt.show()
# # Q2-11 Call a function
# showImageAndOutput(classes, class_image, class_correct, class_total)


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
num_epochs = 20
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

# --------------------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
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

        if (i + 1) % batch_size  == 0:
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

icount = 0
class_image = {}    # Save Input image for visualization
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1, 28*28)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)
    for i in range(batch_size):
        icount += 1
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

        # Q2-11 Save Input image for visualization
        if classes[label] not in class_image:
            class_image[classes[label]] = images[i]


# Q2-11 show 10 classes images with output
def showImageAndOutput(classes, classesimages, correctlist, totallist):
    plotcolumns = 5
    plotrows = int(len(classes) / plotcolumns)

    fig, axes = plt.subplots(nrows=plotrows, ncols=plotcolumns, figsize=(20, 30))
    for ci in range(len(classes)):
        subplot_row = ci // plotcolumns   # plot row
        subplot_col = ci % plotcolumns    # plot column

        axes[subplot_row, subplot_col].text(0, 32, classes[ci], fontsize=20)
        axes[subplot_row, subplot_col].text(0, 36, 'Correct#/Total# : %d / %d' %(correctlist[ci], totallist[ci]), fontsize=15)
        axes[subplot_row, subplot_col].axis('off')
        img = classesimages[classes[ci]].cpu().numpy()
        X = np.reshape(img, (28, 28))
        axes[subplot_row, subplot_col].imshow(X, cmap='gray')

    plt.show()

# Q2-11 Call a function
showImageAndOutput(classes, class_image, class_correct, class_total)

# --------------------------------------------------------------------------------------------
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')


