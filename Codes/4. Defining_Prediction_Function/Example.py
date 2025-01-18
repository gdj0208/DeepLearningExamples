
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn

# < 1. 레이어 함수 정의 > ==========================================

# 1번 선형함수 (입력수 : 784, 출력수 : 128)
l1 = nn.Linear(784, 128)

# 여기서 l1으로 나온 128은 은닉층의 노드 수에 해당한다.

# 2번 선형함수 (입력수 : 128, 출력수 : 10)
l2 = nn.Linear(128, 10)

# 활성화 함수
relu = nn.ReLU(inplace = True)


# < 2. 입력 텐서로부터 출력 텐서를 계산 > ===============================

# 더미 입력 데이터 작성 (100행, 784열의 2계 텐서(행렬))
inputs = torch.randn(100,784)

# 중간 텐서 1 계산
m1 = l1(inputs)

# 중간 텐서 2 계산
m2 = relu(m1)

# 출력 텐서 계산
outputs = l2(m2)

# 입력 텐서와 출력 텐서 shape 확인
print('입력텐서 : ', inputs.shape)
print('출력텐서 : ', outputs.shape)

# 출력 결과 :
# 입력텐서 :  torch.Size([100, 784])
# 출력텐서 :  torch.Size([100, 10])
