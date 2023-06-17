import matplotlib
import pylab
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 数据准备
data_dir = './catdata/cat_12_train/'
test_dir= './catdata/cat_12_test/'
annotations_file = './catdata/train_list.txt'
MODEL_PATH = 'catdata/cat_classification_model.pth'

img_width, img_height = 224, 224

batch_size = 64
num_classes = 12
split_percent = 0.85
learning_rate = 0.01
momentum = 0.9

# 数据转换和增强
data_transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet50()


# 自定义数据集类
class CatDataset(Dataset):
    train_loader: DataLoader
    validate_loader: DataLoader
    test_loader: DataLoader

    def __init__(self, is_test=False, transform=None):
        self.transform = transform
        self.is_test_mode = is_test
        self.imgs, self.labels = self.load_dataset()
        # self.annotations = self.parse_annotations(annotations_file)
        # train_size = int(len(self) * split)
        # test_size = int(len(self) * (1 - split))
        # train_dataset, validate_data_set = torch.utils.data.random_split(self, [train_size, test_size])
        # self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # self.validate_loader = DataLoader(validate_data_set, batch_size=batch_size, shuffle=True)
        # self.test_loader = self.load_test_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.is_test_mode:
            return img, img_path
        else:
            return img, self.labels[index]

    def load_dataset(self):
        images = []
        labels = []
        if self.is_test_mode:
            files = os.listdir(test_dir)
            for img in files:
                images.append(os.path.join(test_dir, img))
        else:
            with open(annotations_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # print(line)
                    img_path, label = line.split("\t")
                    img_path = os.path.join(data_dir, img_path)
                    images.append(img_path)
                    label = int(label)
                    labels.append(label)
        return images, labels


# 加载数据集

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# GPU加速（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train(dataset_loader,num_epochs=10):
    # 训练模型
    losses = []
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataset_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        losses.append(loss.item())
    print(losses)
    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)


def validate(dataset_loader):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    running_accuracy = 0
    total = 0
    is_cuda = False
    if device.type == "cuda":
        is_cuda = True
    with torch.no_grad():
        for data in dataset_loader:
            inputs, labels = data
            if is_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += labels.size(0)
            running_accuracy += (predicted == labels).sum().item()
        print('Accuracy: %.2f %%' % (100 * running_accuracy / total))


def train_and_validate():
    dataset = CatDataset(is_test=False, transform=data_transform)
    train_size = int(len(dataset) * split_percent)
    test_size = len(dataset) - train_size
    train_dataset, validate_data_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_data_set, batch_size=batch_size, shuffle=True)
    train(train_loader)
    validate(validate_loader)


def test():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    test_dataset = CatDataset(is_test=True, transform=data_transform)
    dataset_loader = DataLoader(test_dataset, shuffle=False)
    is_cuda = False
    with torch.no_grad():
        size = len(test_dataset)
        # for i in range(size):
        for data in dataset_loader:
            inputs,image_path = data#dataset_loader[i]
            image_path = image_path[0][10:]
            # image_path = image_path.replace("\.\/catadata\/","")
            if is_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            print(f"{image_path},{predicted.item()}")


#train_and_validate()
test()