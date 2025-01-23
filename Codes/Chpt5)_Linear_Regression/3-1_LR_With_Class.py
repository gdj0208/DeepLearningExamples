
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# < 1. 모델을 정의하는 본질적인 부분 > ==================================
class Net(nn.Module) :
    def __init__(self, n_input, n_output):
        # 부모 클래스인 nn.Module 초기화
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(n_input, n_output)

    # 예츨 함수 정의
    def forward(self, x):
        # 선형 회귀
        x1 = self.l1(x)
        return x1



# < 2. 커스텀 클래스를 활용한 예측 > ====================================

# 더미 입력
inputs = torch.ones(100, 1)

# 인스턴스 생성
n_input = 1
n_output = 1
net = Net(n_input, n_output)

# 예측
outputs = net(inputs)

print(outputs)



# < 3. 손실 함수 정의 및 호출 > ====================================

# (2) 손실 함수 : 평균 제곱 오차
criterion = nn.MSELoss()

# MSE Loss란 평균 제곱 오차를 의미함.

# 손실 계산 : 
# 딥러닝을 위한 수학의 결과와 일치시키기 위해 2로 나눈 값을 손실로 함.
# outputs는 위 <2. 커스텀 클래스를 활용한 예측>에서 도출된 출력이다.
# labels1은 데이터의 실제 차이이다. 
loss = criterion(outputs, labels1) / 2.0

# (3) 경사 계산
loss.backward()

