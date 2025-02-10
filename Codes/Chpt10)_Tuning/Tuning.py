
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# 드롭 아웃 실험용 데이터
torch.manual_seed(123)
inputs = torch.randn(1,10)
print(inputs)

# 드롭 아웃 함수 동작 테스트
dropout = nn.Dropout(0.3)

# 훈련 페이즈에서 드롭아웃을 하며 학습
dropout.train()
print(dropout.training)
outputs = dropout(inputs)
print(outputs)

# 검증 페이즈에서 드롭아웃을 하지 않고 검증
dropout.eval()
print(dropout.training)
outputs = dropout(inputs)
print(outputs)

'''
출력 결과 :
tensor([[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969,  0.2093, -0.9724, -0.7550, 0.3239, -0.1085]])
True
tensor([[-0.0000,  0.1719, -0.5280, -0.3435, -1.7099,  0.2990, -0.0000, -1.0786, 0.4627, -0.1550]])
False
tensor([[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969,  0.2093, -0.9724, -0.7550, 0.3239, -0.1085]])
'''

# 무작위로 수평 반전을 하고 랜덤으로 사각형 영역을 삭제함
transfrom_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3), value=0, inplace=True)
])


'''
# < 외부 라브리 가져오기 > ================================

# 공통 함수 불러오기
from pythonlibs.torch_lib1 import *


#공통 함수 확인
print(README)
'''