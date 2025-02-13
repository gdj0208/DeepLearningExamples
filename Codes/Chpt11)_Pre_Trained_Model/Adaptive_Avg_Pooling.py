
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

# bm : base model
import __init__ as bm

# import ssl
# import certifi

# < 0. 적응형 풀링 함수 > ======================================
p = nn.AdaptiveAvgPool2d((1,1))
print(p)

# 선형 함수 정의
l1 = nn.Linear(32, 10)
print(l1)


# < 1. 사전 학습 모델 시뮬레이션 > ===============================
inputs = torch.randn(100, 32, 16, 16)
m1 = p(inputs)
m2 = m1.view(m1.shape[0], -1)
m3 = l1(m2)

print(inputs)
print(m1.shape)
print(m2.shape)
print(m3.shape)


# < 2. 데이터 준비 > ========================================
# transform 정의
# 학습 데이터용
transform_train = transforms.Compose([
    transforms.Resize(112),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(0.5, scale=(0.02,0.33), ratio=(0.3,0.3), value=0, inplace=False)
])

# 학습 데이터용
transform_test = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])


batch_size = 50 # 배치 사이지 지정

# GPU 연산을 위한 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    

data_root = './Codes/Chpt9)_CNN/data'

# 훈련 데이터셋 
train_set = datasets.CIFAR10(
    root = data_root,
    train = True,
    download = True,
    transform = transform_train
)
# 검증 데이터셋 
test_set = datasets.CIFAR10(
    root = data_root,
    train = False,
    download = True,
    transform = transform_test
)

# 훈련용 데이터 로더
train_loader = DataLoader(train_set, batch_size, shuffle=True)
# 검증용 데이터 로더
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# 난수 고정
bm.torch_seed()

# < 3. ResNet-18 불러오기 > ================================
# ssl._create_default_https_context = ssl._create_unverified_context

net= models.resnet18(pretrained=True)
# print(net)


# < 4. 기타 초기화 > =========================================
n_output = 10

fc_in_features = net.fc.in_features             # 최종 레이어 함수 입력 차원수 확인
net.fc = nn.Linear(fc_in_features, n_output)    # 최종 레이어 함수 교체

net = net.to(device)

lr = 0.001                          # 학습률
criterion = nn.CrossEntropyLoss()   # 손실 함수 정의
optimizer = bm.optim.SGD(net.parameters(), lr, momentum=0.9)
history = bm.np.zeros((0,5))

# 학습
num_epoch = 5
history = bm.fit(net, optimizer, criterion, num_epoch, train_loader, test_loader, device, history)

# < 5. 결과 평가 > =============================================
bm.evaluate_history(history)