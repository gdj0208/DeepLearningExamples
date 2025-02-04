
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

# 다운로드 받을 디렉토리 명
data_root = './MNIST_data'

# 데이터 다운로드
'''
# < 1. Transforms를 사용하지 않은 버전 > ================================================
# 데이터를 코드를 활용하여 다운을 받을 수 없어서 인터넷에서 수동으로 검색하여 다운로드하였다.
# 참고 링크 : https://eclipse360.tistory.com/112
# 위 링크에서 직접 파일을 4개를 다운받아서 ./data/MNIST/raw에 다운로드 하였다.

train_set = datasets.MNIST(root=data_root, train=True, download=True)
train_test = datasets.MNIST(root = data_root, train=False)


# 다운로드한 파일 확인
print('데이터 건수 : ', len(train_set))    
image, label = train_set[0]             # 첫 번째 요소 가져오기
print('입력 데이터 타입 : ', type(image))   # 입력 데이터 타입 :  <class 'PIL.Image.Image'>
print('정답 데이터 타입 : ', type(label))   # 정답 데이터 타입 :  <class 'int'>

# 입력 데이터를 이미지로 출력
plt.figure(figsize=(1,1))
plt.title(f'{label}')
plt.imshow(image, cmap='gray_r')
plt.axis('off')
plt.show()

#첫 20개의 데이터를 정답 데이터와 함께 이미지로 출력하는 방법
plt.figure(figsize=(10,3))
for i in range(20):
    ax = plt.subplot(2, 10, i+1)

    # image와 label 취득
    image, label = train_set[i]

    # 이미지 출력
    plt.imshow(image, cmap='gray_r')
    ax.set_title(f'{label}')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# < 2. Transforms를 사용한 버전 > ================================================
transform1 = transforms.Compose([
    transforms.ToTensor(),                  # 데이터를 텐서로 변환
    transforms.Normalize(0.5, 0.5),         # 데이터 정규화
    transforms.Lambda(lambda x:x.view(-1)), # 현재 텐서를 1계 텐서로 변환
    ])


# < 3. Lambda를 적용한 버전 > ================================================
transform1 = transforms.Compose([
    transforms.ToTensor(),                  # 데이터를 텐서로 변환
    transforms.Normalize(0.5, 0.5),         # 데이터 정규화
    transforms.Lambda(lambda x:x.view(-1)), # 현재 텐서를 1계 텐서로 변환
    ])
'''

# 3. Lambda를 적용한 버전
transform1 = transforms.Compose([
    transforms.ToTensor(),                  # 데이터를 텐서로 변환
    transforms.Normalize(0.5, 0.5),         # 데이터 정규화
    transforms.Lambda(lambda x:x.view(-1)), # 현재 텐서를 1계 텐서로 변환
    ])
train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform1)
train_test = datasets.MNIST(root = data_root, train=False, transform=transform1)

# 다운로드한 파일 확인
# print('데이터 건수 : ', len(train_set))    
# image, label = train_set[0]             # 첫 번째 요소 가져오기
# print('입력 데이터 타입 : ', type(image))     # 입력 데이터 타입 : <class 'torch.Tensor'> 
# print('입력 데이터 shape : ', image.shape)   # 입력 데이터 shape :  torch.Size([784])
# print('정답 데이터 타입 : ', type(label))     # 정답 데이터 타입 :  <class 'int'>
# print('최소값 : ', image.data.min())        # 최소값 :  tensor(0.)
# print('최대값 : ', image.data.max())        # 최대값 :  tensor(1.)


# < 데이터로더를 활용한 미니 배치 데이터 생성 > ===================================================

# 미니 배치 사이즈 지정
batch_size = 500

# 훈련용 데이터 로더(훈련용은 셔플을 적용함)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 검증용 데이터 로더
test_loader = DataLoader(train_test, batch_size=batch_size, shuffle=False)

# 데이터 그룹을 몇 개까지 가져올 수 있는지 확인하는 방법
'''
# 몇 개의 그룹으로 데이터를 가져올 수 있는가
print(len(train_loader))


print(images.shape) # torch.Size([500, 784])
print(labels.shape) # torch.Size([500])
'''


# < 모델 정의 > ===========================================================

# 데이터로더로부터 가장 처음 한 세트를 가져옴
for images, labels in train_loader:
    break

# 모델의 입력층/은닉츨/출력층 차원 수 정의
n_input = images.shape[1]                        # 입력차원수
n_output = len(set(list(labels.data.numpy())))   # 출력 차원 수
n_hidden = 128                                  # 은닉층 노드 수

class Net(nn.Module) :
    def __init__(self, n_input, n_output, n_hidden) :
        super().__init__()

        # weight와 bias를 1.0으로 설정하지 않는다.
        # 1.0으로 설정 시 모델의 파라미터 수가 방대한 탓에 학습이 잘 이루어지지 않는다.

        self.l1 = nn.Linear(n_input, n_hidden)  # 은닉층 정의
        self.l2 = nn.Linear(n_hidden, n_output) # 출력층 정의
        self.relu = nn.ReLU(inplace=True)

    # 위 함수에서 x3에 대해서 relu()가 적용되지 않는 것은 이후에 softmax()를 적용할 것이기 때문
    def forward(self, x) :
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3


# < 변수 초기화 > ==============================================================

# 난수 고정
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# GPU 연산을 위한 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

net = Net(n_input, n_output, n_hidden)  # 모델 정의
net = net.to(device)                    # 모델 GPU에 전송

# 기타 요소들 초기화
lr = 0.01                                       # 학습률
criterion = nn.CrossEntropyLoss()               # 손실 함수
optimizer = optim.SGD(net.parameters(), lr)     # 최적화함수 : 경사하강법
num_epoch = 100                                 # 반복 횟수
history = np.zeros((0,5))                       # 평가 결과 기록


# < 경사 하강 > ===========================================================
# 라이브러리 임포트 : from tqdm.notebook import tqdm

for epoch in range(num_epoch) :
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    n_train, n_test = 0,0

    # 훈련 페이즈
    for inputs, labels in tqdm(train_loader):
        n_train += len(labels)

        # 데이터 GPU로 보내기
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 경사 초기화
        optimizer.zero_grad()
        
        outputs = net(inputs)               # 예측 계산
        loss = criterion(outputs, labels)   # 손실 계산
        loss.backward()                     # 경사 계산
        optimizer.step()                    # 파라미터 수정
        
        predicted = torch.max(outputs, 1)[1]  # 예측 라벨 산출

        # 손실과 정확도 계산
        train_loss += loss.item()
        train_acc += (predicted==labels).sum().item()


    # 검증 페이즈
    for inputs_test, labels_test in test_loader :
        n_test += len(labels_test)

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)

        outputs_test = net(inputs_test)                 # 예측 계산
        loss_test = criterion(outputs_test, labels_test)# 손실 계산
        predicted_test = torch.max(outputs_test, 1)[1]  # 예측 라벨 산출

        # 손실과 정확도 계산
        val_loss += loss_test.item()
        val_acc += (predicted_test==labels_test).sum().item()

    print(epoch)

    train_acc /= n_train
    val_acc /= n_test

    if((epoch)% 10 == 0) :
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

# < 결과 확인 > ===========================================================

# 손실율
plt.plot(history[:,0], history[:,1], 'b', label='Train')
plt.plot(history[:,0], history[:,3], 'k', label='Test')
plt.xlabel('Repeat')
plt.ylabel('Loss')
plt.title('Learning Rate(LOSS)')
plt.legend()
plt.show()

# 정확도
plt.plot(history[:,0], history[:,2], 'r', label='Train')
plt.plot(history[:,0], history[:,4], 'g', label='Test')
plt.xlabel('Repeat')
plt.ylabel('Loss')
plt.title('Learning Rate(ACCURACY)')
plt.legend()
plt.show()