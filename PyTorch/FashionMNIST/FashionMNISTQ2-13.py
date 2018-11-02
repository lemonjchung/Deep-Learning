# Q2-13
# Install tensorboard : https://stackoverflow.com/questions/33634008/how-do-i-install-tensorflows-tensorboard
# pip install -U pip
# pip install tensorflow
# pip install tensorboard

# ----- tensorboard --logdir=/users/joanne/Documents/pycharm/ml2-DeepLearning/Pytorch_/Mini_Project/logs/
# ----- http://localhost:6006

# # 1. Log scalar values (Weight and Bias)
# info = {'Bias': torch.mean(net.fc1.bias), 'Weight': torch.mean(net.fc1.weight)}
#
# for tag, value in info.items():
#     logger.scalar_summary(tag, value, i + 1)
#
# # 2. Log values and gradients of the parameters (histogram summary)
# for tag, value in net.named_parameters():
#     tag = tag.replace('.', '/')
#     logger.histo_summary(tag, value.data.cpu().numpy(), i + 1)
#     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), i + 1)
#
# # 3. Log training images (image summary)
# info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}
#
# for tag, images in info.items():
#     logger.image_summary(tag, images, i + 1)
#     logger.histo_summary(tag, images, i + 1)

# TensorBoard Reference : https://github.com/tensorflow/tensorboard/blob/master/README.md#my-tensorboard-isnt-showing-any-data-whats-wrong
# Install tensorboard : https://www.youtube.com/watch?v=GxuY-6l1uDw
# logger.py : https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard

# --------------------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time
from logger import Logger

# --------------------------------------------------------------------------------------------
# Choose the right values for x.
input_size = 28*28
hidden_size = 10
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

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# --------------------------------------------------------------------------------------------
# Choose the right argument for xx
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

#### tensorboard
logger = Logger('./logs')
#### tensorboard

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

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (Weight and Bias)
            info = {'Bias': torch.mean(net.fc1.bias), 'Weight': torch.mean(net.fc1.weight) }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, i + 1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), i + 1)
                logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), i + 1)

            # 3. Log training images (image summary)
            info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}

            for tag, images in info.items():
                logger.image_summary(tag, images, i + 1)
                logger.histo_summary(tag, images, i + 1 )

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
# There is bug here find it and fix it
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