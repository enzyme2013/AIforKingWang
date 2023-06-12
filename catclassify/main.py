import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


# 数据准备
data_dir = './catdata/'
annotations_file = './catdata/train_list.txt'
img_width, img_height = 224, 224
batch_size = 32
num_classes = 12

# 数据转换和增强
data_transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 自定义数据集类
class CatDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = self.parse_annotations(annotations_file)

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
dataset = CatDataset(data_dir, annotations_file, transform=data_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 构建模型
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# GPU加速（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 输出每个epoch的损失
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 保存模型
torch.save(model.state_dict(), 'catdata/cat_classification_model.pth')


