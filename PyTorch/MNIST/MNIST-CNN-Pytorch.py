import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Initialize size
batch_size=100
epoch_size=1
learning_rate = 0.001
target_size=10

# Get train and test data
train_data = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

train_load = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_load = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Convolution Neural Network Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, target_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


# CNN model
cnn = CNN()
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train model
for epoch in range(epoch_size):
    for i, (images, labels) in enumerate(train_load):
        images = Variable(images)
        labels = Variable(labels)

        # model forward, backward, optimizer
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0 :
            print('Epoch: %d/%d Iter: %d/%d Loss: %.4f' %(epoch, epoch_size, i, len(train_data)//batch_size, loss.item()))


# Test Model
cnn.eval()   # change model to eval model
correct=0
total=0
for images, labels in test_load:
    images = Variable(images)
    output = cnn(images)
    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    correct += int((predicted == labels).sum())

print('10000 Test Images Accuracy %d %%' %(100 * (correct/total)))

torch.save(cnn.state_dict(), 'cnn.pkl')

# # visualization image
# import matplotlib.pyplot as plt
# import numpy as np
#
# datatier = iter(test_load)
# images, labels = datatier.next()
# images.numpy()
# output = cnn(images)
# _, predicted = torch.max(output,1)
# preds = np.squeeze(predicted.numpy())
#
# fig = plt.figure(figsize=(25,4))
# for i in range(target_size):
#     ax = fig.add_subplot(2, target_size/2, i+1)
#     img = images[i] / 2 + 0.5  # unnormalize
#     plt.imshow(np.transpose(img, (1, 2, 0)))
# plt.show()
