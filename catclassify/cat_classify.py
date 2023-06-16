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
data_dir = './catdata/'
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


model = models.resnet50(pretrained=True)


# 自定义数据集类
class CatDataset(Dataset):
    train_loader:DataLoader
    validate_loader:DataLoader
    def __init__(self, data_dir, annotations_file, split=0.85, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = self.parse_annotations(annotations_file)
        train_size = int(len(self) * split)
        test_size = int(len(self) * (1 - split))
        train_dataset, validate_data_set = torch.utils.data.random_split(self, [train_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.validate_loader = DataLoader(validate_data_set,batch_size=batch_size,shuffle=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, label = self.annotations[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def parse_annotations(self, annotations_file):
        annotations = []
        with open(annotations_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # print(line)
                img_path, label = line.split("\t")
                img_path = os.path.join(self.data_dir, img_path)
                # print(img_path)
                label = int(label)
                annotations.append((img_path, label))
        return annotations



# 加载数据集

dataset = CatDataset(data_dir, annotations_file, split=split_percent,transform=data_transform)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# GPU加速（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train(num_epochs = 10):
    # 训练模型
    losses = []
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataset.train_loader:
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


def validate():
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    running_accuracy = 0
    total = 0
    is_cuda = False
    if device.type == "cuda":
        is_cuda = True
    with torch.no_grad():
        for data in dataset.validate_loader:
            inputs, labels = data
            if is_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # outputs = outputs.to(torch.int)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += labels.size(0)
            running_accuracy += (predicted == labels).sum().item()
        print('Accuracy: %.2f %%' % (100 * running_accuracy / total))


def test_pic(img_path):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # img = np.asarray(Image.open(img_path))
    # imgplot = plt.imshow(img)
    # pylab.show()
    img = data_transform(Image.open(img_path))
    if device.type == "cuda":
        img = img.cuda()
    predict_output = model(img.unsqueeze(0))
    _, predict = torch.max(predict_output,1)
    #

    print(predict)


train(30)
validate()
# matplotlib.use('TkAgg')
# test_pic("./catdata/cat_12_test/0inVXMEgaBO4Fcrhdj9bkLvHzN71yTuI.jpg")