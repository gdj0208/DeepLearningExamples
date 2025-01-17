
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# < 1. 선형함수 정의 예제 > ==========================================

# 선형함수 정의 (입력 텐서의 차원수 : 2, 출력 텐서의 차원수 : 3)
l3 = nn.Linear(2,3)
print(l3)
# 출력 결과 : Linear(in_features=2, out_features=3, bias=True)
# in_features : 입력 텐서 차원수
# out_features : 출력 텐서 차원수
# bias : 함수에 정수항을 포함할 것인지에 관한 옵션

# 인스턴스 변수의 내부 파라미터 확인
for para in l3.named_parameters():
    print('name : ', para[0])
    print('tensor : ', para[1]);


# < 2. 1입력, 1출력 예제 > ==========================================

# 선형함수 정의
l1 = nn.Linear(1,1)

# 변수 선언
x_np = np.arange(-2, 2.1, 0.25)

# 배열을 텐서로 변환시 float() 적용.
# float() 미사용시 double타입이 되어 선형함수 호출시 에러를 발생시킴
x = torch.tensor(x_np).float()

# x의 사이즈를 (N, 1)로 변경
# weight와의 연산을 위해 스칼라가 아닌 [5,1]과 같은 2차원 텐서로 전환
x = x.view(-1,1)

# 결과 확인
print(x)

# 출력 결과 
# tensor([[-2.0000], [-1.7500], .....[2.0000]])


# 예측값 도출
y = l1(x)

# 예측값 출력
print(y.data)


# < 3. 2입력, 1출력 예제 > ==========================================

# 선형함수 정의
l2 = nn.Linear(2,1)

# 선형함수의 weight, bias 정의
nn.init.constant_(l2.weight, 1.0)
nn.init.constant_(l2.bias, 2.0)
# y = (1)x1 + (1)x2 + 2로 설정

# 변수 선언
x_np = np.array([[0,0], [0,1], [1,0], [1,0]])
x = torch.tensor(x_np).float()

# 결과 확인
print(x)

# 예측값 도출
y = l2(x)

# 예측값 출력
print(y.data)


# < 4. 2입력, 3출력 예제 > ==========================================

# 선형함수 정의
l3 = nn.Linear(2,3)

# 선형함수의 weight, bias 정의
nn.init.constant_(l3.weight[0,:], 1.0)
nn.init.constant_(l3.weight[1,:], 2.0)
nn.init.constant_(l3.weight[2,:], 3.0)
nn.init.constant_(l3.bias, 2.0)
# 파라미터는 다음과 같이 설정된 상태
# [1, 1]
# [2, 2]
# [3, 3]

# 변수 선언
x_np = np.array([[0,0], [0,1], [1,0], [1,0]])
x = torch.tensor(x_np).float()

# 결과 확인
print(x)

# 예측값 도출
y = l3(x)

# 예측값 출력
# [4,2]벡터(x)에 [2,3]벡터(파라미터)를 곱했으니 [4,3]벡터가 나올 것
print(y.shape)
print(y.data)