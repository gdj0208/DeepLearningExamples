

import torch
import torchvision.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

import Base_Functions

# <0. 2_Base_Fuction.py의 함수들 >

# < 1. Transforms 정의 > ======================================
# transform1의 1계 텐서화
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.Lambda(lambda x : x.view(-1)),
])

# transform2 정규화만 실시
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])



# 훈련 데이터셋 (1계 텐서 버전)
train_set1 = datasets.CIFAR10(
    root = data_root,
    train = True,
    download = True,
    transform = transform1
)
# 검증 데이터셋 (1계 텐서 버전)
test_set1 = datasets.CIFAR10(
    root = data_root,
    train = False,
    download = True,
    transform = transform1
)

# 훈련 데이터셋 (3계 텐서 버전)
train_set2 = datasets.CIFAR10(
    root = data_root,
    train = True,
    download = True,
    transform = transform2
)
# 검증 데이터셋 (3계 텐서 버전)
test_set2 = datasets.CIFAR10(
    root = data_root,
    train = False,
    download = True,
    transform = transform2
)

'''
# 데이터 다운로드 확인
image1, label1 = train_set1[0]
image2, label2 = train_set2[0]

print(image1.shape)     # torch.Size([3072])
print(image2.shape)     # torch.Size([3, 32, 32])
'''

# < 3. 데이터로더 정의 > =======================================
batch_size = 100

# 훈련용 데이터로더 (훈련용이므로 셔플 = True)
train_loader1 = DataLoader(train_set1, batch_size=batch_size, shuffle=True)
train_loader2 = DataLoader(train_set2, batch_size=batch_size, shuffle=True)
# 검증용 데이터로더 (검증용이므로 셔플=False)
test_loader1 = DataLoader(test_set1, batch_size=batch_size, shuffle=False)
test_loader2 = DataLoader(test_set2, batch_size=batch_size, shuffle=False)

'''
# 데이터로더 확인
for images1, labels1 in train_loader1:
    break
for images2, labels2 in train_loader2:
    break

# 각 데이터 shape 확인
print(images1.shape)    # torch.Size([100, 3072])
print(images2.shape)    # torch.Size([100, 3, 32, 32])
'''

# < 4. 검증데이터 이미지 표시 > ==================================
# 정답 라벨 정의
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 검증 데이터의 첫 50개 출력
# show_images_labels(test_loader2, classes)

# < 5. 모델 정의 > ==================================
for images, labels in train_loader1:
    break

n_input = images.shape[1]                       # 입력 차원수 : 3*32*32 = 3072
n_output = len(set(list(labels.data.numpy())))  # 출력 차원 수 (분류 클래스의 수 = 10)
n_hidden = 128                                  # 은닉층 노드 수

# 전결합형 모델
class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        self.l1 = nn.Linear(n_input, n_hidden)  # 은닉층 정의
        self.l2 = nn.Linear(n_hidden, n_output) # 출력층 정의
        self.relu = nn.ReLU(inplace=True)       # relu 정의

    def forward(self, x):
         x1 = self.l1(x)
         x2 = self.relu(x1)
         x3 = self.l2(x2)
         return x3

'''
# CNN 모델
class CNN(nn.Module):
    def __init__(self, n_output, n_hidden):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)  
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(6272, n_hidden)         # 6227은 conv&pool.py에서 4.1계화 함수에서 구해졌다.
        self.l2 = nn.Linear(n_hidden, n_output)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool
        )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self, x):
         x1 = self.features(x)
         x2 = self.flatten(x1)
         x3 = self.classifier(x2)
         return x3
'''

# CNN 모델 (개선)
class CNN(nn.Module):
    def __init__(self, n_output, n_hidden):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)  
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(6272, n_hidden)         # 6227은 conv&pool.py에서 4.1계화 함수에서 구해졌다.
        self.l2 = nn.Linear(n_hidden, n_hidden)     # 추가 코드 (은닉층 추가)
        self.l3 = nn.Linear(n_hidden, n_output)
        
        self.dropout = nn.Dropout(0.3)              # 추가 코드 (0.5의 확률로 드롭아웃)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool
        )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.dropout,   # 추가코드
            self.l2,
            self.relu,
            self.dropout,   # 추가코드
            self.l3
        )

    def forward(self, x):
         x1 = self.features(x)
         x2 = self.flatten(x1)
         x3 = self.classifier(x2)
         return x3

# 기타 초기화
Base_Functions.torch_seed()        # Seed값 초기화

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    # GPU 연산을 위한 설정

# 모델 설정
# net = Net(n_input, n_output, n_hidden).to(device)   # 전결합형 모델용 코드
net = CNN(n_output, n_hidden).to(device)            # CNN 모델용 코드 

criterion = nn.CrossEntropyLoss()                   # 손실 함수
lr = 0.01                                           # 학습률
optimizer = optim.SGD(net.parameters(), lr)         # 최적화 함수 (경사하강법)
num_epoch = 80                                      # 반복 횟수
history = np.zeros((0,5))                           # 평가 결과 기록

# 모델 학습
# history = Base_Functions.fit(net, optimizer, criterion, num_epoch, train_loader1, test_loader1, device, history)  # 전결합형 모델용 코드
history = Base_Functions.fit(net, optimizer, criterion, num_epoch, train_loader2, test_loader2, device, history)    # CNN 모델용 코드

# 손실과 정확도 출력
Base_Functions.print_loss_acc(history)

