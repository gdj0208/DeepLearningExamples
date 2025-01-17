
import numpy as np
import pandas as pd  # Pandas 라이브러리 추가
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os


# < 1. 변수 정의 > ================================================

# 변수 정의
n_input = x.shape[1]

# 출력 차원수
n_output = 1

# 입출력 차원수 출력
# 1입력 1출력
print('입력 차원수 : ', n_input)
print('출력 차원수 : ', n_input)



# < 2. 변수 정의 > ================================================
class Net(nn.modules) :
    def __init__(self, n_input, n_output) :

        # 부모 클래스 nn.Module의 초기화 호출
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(n_input, n_output)

        #초기값을 모두 1롤 설정
        nn.init.constant_(self.l1.weight, 1.0)
        nn.init.constant_(self.l1.bias, 1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1



# < 3. 인스턴스 생성 > ================================================

# Net 클래스의 인스턴스인 net 변수를 정의
# 입력 텐서 inputs로부터 예측값인 출력텐서 outputs를 출력
net = Net(n_input, n_output)
outputs = net(inputs)



# < 4. 모델 내부의 변수값 표시 > ================================================

# 모델 안의 파라미터를 확인
# 모델 안의 변수를 가져오기 위해 named_parameters 함수를 사용

# 첫 번째 요소 : 변수명
# 두 번째 요소 : 변수값

for parameter in net.named_parameters():
    print('변수명 : {parameter[0]}')
    print('변수값 : {parameter[1].data}')



# < 5. parameters 함수의 호출 > ================================================

# named_parameters() : {}'파라미터 명칭', '파라미터 값'}의 쌍을 반환
# parameters() : '파라미터 값'만을 반환

for parameter in net.parameters() :
    print(parameter)



# < 6. 모델 개요 표시 > ================================================

# 모델 개요 표시 1
print(net)

# 모델 개요 표시 2
# torchinfo라는 전용 라이브러리 이용
from torchinfo import summary

summary(net, (1,))



# < 7. 손실 함수와 최적화 함수의 정의 > ================================================

# 손실함수 : 평균 제곱 오차
criterion = nn.MSELoss()

# 학습률
lr = 0.01

# 최적화 함수 : 경사하강법
optimizer = optim.SGD(net.parameters(), lr = lr)
#optim : Pytorch에서 제공하는 최적화 알고리즘 모듈
#        파라미터를 업데이트하기 위해 사용
#        torch.optim 라이브러리 필요

# SGD : 경사하강법 (torch.optim 내의 최적화 알고리즘중 하나)