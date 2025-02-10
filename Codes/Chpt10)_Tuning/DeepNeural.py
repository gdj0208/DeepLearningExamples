
from pythonlibs.torch_lib1 import *

# 모델
class CNN(nn.Module) :
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))
        
        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, num_classes)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
    
        self.features = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu, self.maxpool, self.dropout1,
            self.conv3, self.bn3, self.relu, self.conv4, self.bn4, self.relu, self.maxpool, self.dropout2,
            self.conv5, self.bn5, self.relu, self.conv6, self.bn6, self.relu, self.maxpool, self.dropout3,
        )

        self.classifier = nn.Sequential(
            self.l1, self.relu, self.dropout3, self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3

# < 1. Transforms 정의 > ======================================
# transform1의 1계 텐서화
transform2 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])


# < 2. 데이터셋 정의 > ======================================
data_root = './Codes/Chpt9)_CNN/data'

# 훈련 데이터셋 (3계 텐서 버전)
train_set = datasets.CIFAR10(
    root = data_root,
    train = True,
    download = True,
    transform = transform2
)
# 검증 데이터셋 (3계 텐서 버전)
test_set = datasets.CIFAR10(
    root = data_root,
    train = False,
    download = True,
    transform = transform2
)


# < 3. 데이터로더 정의 > =======================================
batch_size = 100

# 훈련 데이터셋 (3계 텐서 버전)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# 검증 데이터셋 (3계 텐서 버전)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# < 4. 검증데이터 이미지 표시 > ==================================
# 정답 라벨 정의
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# < 5. 모델 정의 > ==================================
for images, labels in train_loader:
    break

n_input = images.shape[1]                       # 입력 차원수 : 3*32*32 = 3072
n_output = len(set(list(labels.data.numpy())))  # 출력 차원 수 (분류 클래스의 수 = 10)
n_hidden = 128                                  # 은닉층 노드 수


# < 6. 기타 초기화 > ==================================
torch_seed()                            # 난수 고정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    # GPU 연산을 위한 설정

# 모델 인스턴스 생성
net = CNN(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history = np.zeros((0,5))


# < 6. 기타 초기화 > ==================================
num_epochs = 50
history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)

evaluate_history(history)