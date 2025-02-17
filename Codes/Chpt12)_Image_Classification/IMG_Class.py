
from library.torch_lib1 import *
from torchvision import models
import os
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
urllib.request.urlretrieve("https://google.com", "test.html")  # SSL 테스트

# transform 정의
test_transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])
''' 
원본 이미지의 화소가 (400,200)일 경우, Reszie(224)의 결과는 (448,224)가 된다.
(224, 224)가 되기 위해서는 CenterCrop() 처리가 필요
'''

# 데이터셋 정의)
data_dir = 'Codes/Chpt12)_Image_Classification/library/images/dog_wolf'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

classes = ['dog', 'wolf']
train_data_1 = datasets.ImageFolder(train_dir, transform=train_transform)
train_data_2 = datasets.ImageFolder(train_dir, transform=train_transform)
test_data = datasets.ImageFolder(test_dir,transform=train_transform)


# 데이터로더 정의
batch_size = 5
train_loader_1 = DataLoader(train_data_1, batch_size, shuffle=True)
train_loader_2 = DataLoader(train_data_2, batch_size=40, shuffle=False)
test_loader_1 = DataLoader(test_data, batch_size, shuffle=False)
test_loader_2 = DataLoader(test_data, batch_size=10, shuffle=True)

# show_images_labels(train_loader_2, classes, None, None)


# < 기타 변수 초기화 > ================================
# 사전학습 모델 불러오기
net = models.vgg19_bn(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

# GPU 연산을 위한 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    

torch_seed()    # 난수 고정

# 마지막 노드의 출력을 2로 변경
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

net.avgpool = nn.Identity()     # adaptiveAvgPool2d 함수 제거
net = net.to(device)            # GPU 사용

lr = 0.001                          # 학습률
criterion = nn.CrossEntropyLoss()   # 손실 함수 정의
optimizer = optim.SGD(net.classifier[6].parameters(), lr, momentum=0.9)
history = np.zeros((0,5))
num_epoch = 20
history = fit(net, optimizer, criterion, num_epoch, train_loader_1, test_loader_1, device, history)

# 결과 예측
evaluate_history(history)
