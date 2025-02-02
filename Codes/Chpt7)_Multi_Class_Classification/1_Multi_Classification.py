
# 라이브러리 임포트
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# < 1. 데이터 준비 > =============================================

# < 데이터 불러오기 >
iris = load_iris()

# 입력 데이터와 정답 데이터
x_org, y_org = iris.data, iris.target

# 입력 데이터로 sepal(꽃받침)(length(0))와 potal(꽃잎)(length(2))으로 추출
x_select = x_org[:, [0,2]]

# 결과 확인
# print('원본 데이터', x_select.shape, y_org.shape)
# 출력 결과 : 원본 데이터 (150, 2) (150,)


# < 훈련 데이터와 검증 데이터 분할 >
x_train, x_test, y_train, y_test = train_test_split(
    x_select, y_org, train_size=75, test_size=75, random_state=123
)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# 출력 결과 : (75, 2) (75,) (75, 2) (75,)

'''
# < 산포도 출력 >
x_t0 = x_train[y_train == 0]    # 정답이 0인 데이터
x_t1 = x_train[y_train == 1]    # 정답이 1인 데이터
x_t2 = x_train[y_train == 2]    # 정답이 1인 데이터

plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='k', s=50, label='0 (setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='b', s=50, label='1 (versicolor)')
plt.scatter(x_t2[:,0], x_t2[:,1], marker='+', c='k', s=50, label='2 (virginica)')
plt.xlabel('sepal_length')
plt.ylabel('petal_length')
plt.legend()
plt.show()
'''


# < 2. 모델 정의 > =============================================

# < 모델 설정 >
class Net(nn.Module) :
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

        # 초기값을 모두 1ㄹ로 설정
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1
    
# < 학습용 파라미터 설정 >
n_input = x_train.shape[1]          # 입력 차원수 (2)
n_output = len(list(set(y_train)))  # 출력 차원수 (3)

net = Net(n_input, n_output)        # 모델 선언

criterion = nn.CrossEntropyLoss()           # 손실함수 정의
# CrossEntropyLoss()에서 { softmax(), log(), 정답 요소 추출 }을 모두 실행
lr = 0.01                                   # 학습률
optimizer = optim.SGD(net.parameters(), lr) # 최적화 함수 : 경사하강법
num_epoch = 10000                           # 반복 횟수
history = np.zeros((0,5))                   # 평가 결과 기록

# 학습 데이터와 검증 데이터 텐서화
inputs_train = torch.tensor(x_train).float()
outputs_train = torch.tensor(y_train).long()

inputs_test = torch.tensor(x_test).float()
outputs_test = torch.tensor(y_test).long()


# < 3. 학습 > =============================================
for epoch in range(num_epoch) :
    # < 훈련 페이스 >
    optimizer.zero_grad()                           # 경사 초기화
    outputs_pred = net(inputs_train)                # 예측 계산
    loss = criterion(outputs_pred, outputs_train)   # 손실 계산

    loss.backward()                                 # 경사 계산
    optimizer.step()                                # 파라미터 수정
    
    predicted = torch.max(outputs_pred, 1)[1]       # 예측 라벨 산출

    # 손실과 정확도 계산
    train_loss = loss.item()
    train_acc = (predicted == outputs_train).sum() / len(outputs_train)

    
    # < 예측 페이스 >
    outputs_test_pred = net(inputs_test)                    # 예측 계산
    loss_test = criterion(outputs_test_pred, outputs_test)  # 손실 계산
    predicted_test = torch.max(outputs_test_pred, 1)[1]     # 예측 라벨 산출

    # 손실과 정확도 계산
    val_loss = loss_test.item()
    val_acc = (predicted_test == outputs_test).sum() / len(outputs_test)

    if((epoch)% 10 == 0) :
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

# < 4. 결과 확인 > =============================================

# 손실율 확인
plt.plot(history[:,0], history[:,1], 'b', label='train')
plt.plot(history[:,0], history[:,3], 'k', label='test')
plt.xlabel('repeat')
plt.ylabel('loss')
plt.legend()
plt.show()

# 정확도 확인
plt.plot(history[:,0], history[:,2], 'b', label='train')
plt.plot(history[:,0], history[:,4], 'k', label='test')
plt.xlabel('repeat')
plt.ylabel('accuracy')
plt.legend()
plt.show()

'''
# < 5. 번외 - 모델의 출력 확인하기 > =================================

# 정답이 0,1,2인 데이터  :
print(outputs_train[[0,2,3]])   # tensor([1, 0, 2])

# 0,2,3번 데이터의 입력값들 출력  
items = inputs_train[[0,2,3], :]
print(items.data.numpy())
#[[6.3 4.7]
# [5.  1.6]
# [6.4 5.6]]

# softmax()를 적용한 결과 출력
softmax = torch.nn.Softmax(dim=1)
sums = net(items)                   # 각 입력 노드들로 부터 온 값들의 합
softed = softmax(sums)              # 윗 결과에 softmax()를 적용한 겨로가
print(sums.data.numpy())
#[[ 8.807071   14.193753   12.998573  ]
# [12.826236    9.799995    0.17344084]
# [ 6.7954125  15.092808   17.111107  ]]
print(softed.data.numpy())
#[[3.5014073e-03 7.6497829e-01 2.3152030e-01]    # [1]의 값이 가장 크기에 1을 정답으로 함
# [9.5374262e-01 4.6254259e-02 3.0506487e-06]    # [0]의 값이 가장 크기에 0을 정답으로 함
# [2.9224935e-05 1.1729155e-01 8.8267922e-01]]   # [2]의 값이 가장 크기에 2를 정답으로 함
 '''