
import torch
import torch.nn as nn


# < 1. 합성곱과 풀링 기본 > ==========================================
conv1 = nn.Conv2d(3, 32, 3)     
relu = nn.ReLU(inplace = True)
conv2 = nn.Conv2d(32, 32, 3)
# 입력 단자 : ( 입력 채널 수, 출력 채낼 수, 커널 수 )
# 커널 수는 작은 정사각형 한 변의 화소 수를 의미한다

maxpool = nn.MaxPool2d((2,2))
# 인력 단자 : ( 가로 화소 수, 세로 화소 수 )


# conv1의 내부 파라미터
print(conv1.weight.shape)   # torch.Size([32, 3, 3, 3])
print(conv1.bias.shape)     # torch.Size([32])

# conv2의 내부 파라미터
print(conv2.weight.shape)   # torch.Size([32, 3, 3, 3])
print(conv2.bias.shape)     # torch.Size([32])


# < 2. 합성곱과 풀링을 통한 레이어 변경 > ==========================================
# 더미로 입력과 같은 사이즈를 갖는 텐서를 생성
inputs = torch.randn(100, 3, 32, 32)

# CNN 전반부 처리 시뮬레이션
x1 = conv1(inputs)
x2 = relu(x1)
x3 = conv2(x2)
x4 = relu(x3)
x5 = maxpool(x4)

# 각 변수의 shpae 확인      (데이터 건수, 채널 수, 화소(가로), 화소(세로))
print(inputs.shape)     # torch.Size([100, 3, 32, 32])
print(x1.shape)         # torch.Size([100, 32, 30, 30])
print(x2.shape)         # torch.Size([100, 32, 30, 30])
print(x3.shape)         # torch.Size([100, 32, 28, 28])
print(x4.shape)         # torch.Size([100, 32, 28, 28])
print(x5.shape)         # torch.Size([100, 32, 14, 14])


# < 3. nn.Sequentail > ============================================
# 함수 정의
features = nn.Sequential(
    conv1,
    relu,
    conv2,
    relu,
    maxpool
)

outputs = features(inputs)
print(outputs.shape)        # torch.Size([100, 32, 14, 14])


# < 4. 1계화 함수 > =================================================

flatten = nn.Flatten()      # 함수 정의
outputs2 = flatten(outputs) # 동작 테스트

# 결과 확인
print(outputs.shape)        # torch.Size([100, 32, 14, 14])
print(outputs2.shape)       # torch.Size([100, 6272])
