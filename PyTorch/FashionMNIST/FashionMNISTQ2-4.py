# Q2-4
# ====== Deep Network : 3 Hidden Layers and 500 # of Neurons  ======================
# ====== Training Time 29.873754
# Accuracy of the network on the 10000 test images: 87 %

# ====== Shallow Network : 1 Hidden Layers and 10 # of Neurons  ======================
# ====== Training Time 28.275435
# Accuracy of the network on the 10000 test images: 83 %


# ====== Training Time 30.851493
# Accuracy of the network on the 10000 test images: 82 %
# Predicted:  Shirt Dress Shirt   Bag
# Accuracy of Tshirt : 80 %
# Accuracy of Trouser : 98 %
# Accuracy of Pullover : 87 %
# Accuracy of Dress : 91 %
# Accuracy of  Coat : 60 %
# Accuracy of Sandal : 84 %
# Accuracy of Shirt : 35 %
# Accuracy of Sneaker : 90 %
# Accuracy of   Bag : 98 %
# Accuracy of Ankleboot : 96 %

# ====== Deep Network : 3 Hidden Layers and 10 # of Neurons  ======================
# ====== Training Time 27.545849
# Accuracy of the network on the 10000 test images: 83 %
# Predicted:  Shirt Dress Shirt   Bag
# Accuracy of Tshirt : 80 %
# Accuracy of Trouser : 98 %
# Accuracy of Pullover : 76 %
# Accuracy of Dress : 97 %
# Accuracy of  Coat : 69 %
# Accuracy of Sandal : 87 %
# Accuracy of Shirt : 45 %
# Accuracy of Sneaker : 88 %
# Accuracy of   Bag : 96 %
# Accuracy of Ankleboot : 100 %

# ++++++++++++++++++ old ++++++++++++++++++++++++++++++++++++++++++
# ====== Deep Network : hidden layers: 500   ======================
# ====== Training Time 27.804033
# Accuracy of the network on the 10000 test images: 86 %
# Predicted:  Shirt Dress Tshirt   Bag
# Accuracy of Tshirt : 85 %
# Accuracy of Trouser : 96 %
# Accuracy of Pullover : 84 %
# Accuracy of Dress : 94 %
# Accuracy of  Coat : 86 %
# Accuracy of Sandal : 89 %
# Accuracy of Shirt : 41 %
# Accuracy of Sneaker : 90 %
# Accuracy of   Bag : 98 %
# Accuracy of Ankleboot : 100 %

# ====== Shallow Network : hidden layers: 3   =======================
# ====== Training Time 27.352746
# Accuracy of the network on the 10000 test images: 60 %
# Predicted:  Tshirt Dress Pullover   Bag
# Accuracy of Tshirt : 71 %
# Accuracy of Trouser : 94 %
# Accuracy of Pullover : 69 %
# Accuracy of Dress : 61 %
# Accuracy of  Coat : 41 %
# Accuracy of Sandal : 66 %
# Accuracy of Shirt :  9 %
# Accuracy of Sneaker :  0 %
# Accuracy of   Bag : 82 %
# Accuracy of Ankleboot : 100 %


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
hidden_size1 = 500
hidden_size2 = 500
hidden_size3 = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

torch.manual_seed(9000)
# --------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------
train_set = torchvision.datasets.FashionMNIST(root='./data_fashion', train=True, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data_fashion', train=False, download=True, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# # Find the right classes name. Save it as a tuple of size 10.
classes =("Tshirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankleboot")


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
    # ----------- Q2-4 3 Hidden Layer
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()         # nn.LeakyReLU()        # nn.ELU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu = nn.ReLU()         # nn.LeakyReLU()        # nn.ELU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu = nn.ReLU()         # nn.LeakyReLU()        # nn.ELU()
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        return out

# --------------------------------------------------------------------------------------------
# ----------- Q2-4 3 Hidden Layer
net = Net(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
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