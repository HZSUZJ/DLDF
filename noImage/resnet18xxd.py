import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 200
batch_size = 64
learning_rate = 0.0001


class singleResDataset(torch.utils.data.Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        inputs = data.reshape(1, 22, 21)
        label = self.labels[idx]
        return inputs, label


data = np.array(pd.read_csv('freshness.csv').values[:, 1:463])
label = np.array(pd.read_csv('freshness.csv').values[:, 0])
train_datas, test_datas, train_labels, test_labels = train_test_split(data, label, test_size=0.3, random_state=1)

train_dataset = singleResDataset(datas=train_datas,
                                 labels=train_labels)

valid_dataset = singleResDataset(datas=test_datas,
                                 labels=test_labels)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=False, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=64, shuffle=False, drop_last=False
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_image(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet_image, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_image():
    return ResNet_image(BasicBlock, [2, 2, 2, 2])


model = ResNet18_image().to(device)
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
total_step = len(train_loader)
for epoch in range(num_epochs):
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long)
        images = images.float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            images = images.float()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            images = images.float()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        images = images.float()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    y_true = []
    y_pre = []
    for images, labels in test_loader:
        images = images.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long)
        images = images.float()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_pre.append(predicted.cpu())
        y_true.append(labels.cpu())

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
