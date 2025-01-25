
# 라이브러리 임포트
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# < 1. 데이터 호출 > =============================================

# 데이터 불러오기
iris = load_iris()

# 입력 데이터와 정답 데이터
x_org, y_org = iris.data, iris.target

# 결과 확인
print('원본 데이터', x_org.shape, y_org.shape)
# 출력 결과 : 원본 데이터 (150, 4) (150,)



# < 2. 데이터 추출 > =============================================

# 데이터 추출
x_data = iris.data[:100, :2]    # 100행 2열의 내용들을 추출
y_data = iris.target[:100]      # 100행의 내용들을 추출

# 결과 확인
#print(f'x : {x_data}')
#print(f'y : {y_data}')
print('대상 데이터', x_data.shape, y_data.shape)



# < 3. 훈련 데이터와 검증 데이터 분할 > ===============================

# 원본 데이터의 사이즈
print(x_data.shape, y_data.shape)

# 훈련 데이터와 검증 데이터로 분할 & 셔플
'''
- 라이브러리 임포트 필요
from sklearn.model_selection import train_test_split
# 상단에 넣어놓음
'''

# 훈련 데이터는 70개, 검증 데이터는 30개로 설정
'''
- 순서를 바꾸지 않고 분할 시 데이터에 편향이 생길 수 있다.
- 이를 방지하기 위해 train_test_split함수로 데이터를 잘 섞은 후 분할한다.
- random_state : 파라미터 지정시 데이터를 섞을 때 난수 시드가 고정되어 분할 결과는 항상 일치한다.
'''
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size=30, random_state=123
)



'''
# < 4. 산포도 출력 > =============================================
x_t0 = x_train[y_train == 0]    # 정답이 0인 데이터
x_t1 = x_train[y_train == 1]    # 정답이 1인 데이터

plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='b', label='0 (Setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='k', label='1 (Versicolor)')
plt.xlabel('sepal_length')
plt.ylabel('sepla_width')
plt.legend()
plt.show()
'''



# < 4. 모델 정의 > ==============================================
# 입출력 차원수 정의
n_input = x_train.shape[1]  # 입력 차원수 정의 (현 데이터의 경우 2)
n_output = 1                # 출력 차운수 정의 (현 데이터의 경우 1)

# 모델 정의
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        self.sigmoid = nn.Sigmoid()

        # 초기값을 1로 정의
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    # 예측 함수 정의
    def forward(self, x):
        '''
        - 입력 텐서를 선형 함수에 적용한 결과 x1을 생성
        - x1에 시그모이드 함수를 다시 적용한 결과 x2를 생성
        '''
        x1 = self.l1(x)     # 선형함수에 입력값을 넣고 계산한 결과
        x2 = self.sigmoid(x1)    # 계산 결과에 시그모이드 함수를 적용
        return x2



# < 5. 경사 하강 > ==============================================

# < 5-1. 학습 데이터 텐서화 > 
# 입력데이터 와 정답데이터의 텐서화
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()

# 정답 데이터 N:1 차원으로 전환
labels = labels.view((-1, 1))


# < 5-2. 검증 데이터 텐서화 > 
# 검증 데이터 덴서화
inputs_tests = torch.tensor(x_test).float()
labels_tests = torch.tensor(y_test).float()

# 검증용 정답 데이터 N:1 차원으로 전환
labels_tests = labels_tests.view((-1, 1))



# < 6. 초기화 처리 > =============================================
# 학습률 
lr = 0.001

# 초기화
net = Net(n_input, n_output)

# 손실함수 
criterion = nn.BCELoss()

# 최적화 함수 : 경사 하강법
optimizer = optim.SGD(net.parameters(), lr=lr)

# 반복 처리
num_epochs = 10000

# 기록용 리스트 초기화
history = np.zeros((0,5))


# < 7. 메인 루프 > =============================================
for epoch in range(num_epochs) :
    # < 훈련 페이스 >
    # 경사값 초기화
    optimizer.zero_grad()
   
    # 예측 계산 
    outputs = net(inputs)

    # 손실 계산
    loss = criterion(outputs, labels)
    
    # 경사 계산
    loss.backward()

    # 파라미터 수정
    optimizer.step()

    # 손실 저장 (스칼라값 취득)
    train_loss = loss.item()

    # 예측 라벨(1 또는 0 계산)
    predicted = torch.where(outputs < 0.5, 0, 1)

    # 정확도 계산
    train_acc = (predicted == labels).sum() / len(y_train)



    # < 예측 페이스 >
    # 예측 계산
    outputs_test = net(inputs_tests)

    # 손실 계산
    loss_test = criterion(outputs_test, labels_tests)

    # 손실 저장 (스칼라값 취득)
    var_loss = loss_test.item()

    # 예측 라벨 계산
    predicted_test = torch.where(outputs_test < 0.5, 0, 1)
    


    # < 정확도 계산 >
    var_acc = (predicted_test == labels_tests).sum() / len(y_test)

    if(epoch % 10 == 0) :
        item = np.array([epoch, train_loss, train_acc, var_loss, var_acc])
        history = np.vstack((history, item))




# < 8. 결과 확인 > =============================================

# < 학습 곡선 (손실) > 
plt.plot(history[:,0], history[:,1], 'b', label='Train')
plt.plot(history[:,0], history[:,3], 'k', label='Test')
plt.xlabel('Repeat')
plt.ylabel('Loss')
plt.title("Learning Rate (Loss)")
plt.legend()
plt.show()


# < 학습 곡선 (정확도) > 
plt.plot(history[:,0], history[:,2], 'b', label='Train')
plt.plot(history[:,0], history[:,4], 'k', label='Test')
plt.xlabel('Repeat')
plt.ylabel('Accuracy')
plt.title("Learning Rate (Accuracy)")
plt.legend()
plt.show()


# < 로지스틱 회귀 모델의 결정 경계 표시 >
x_t0 = x_train[y_train == 0]    # 정답이 0인 데이터
x_t1 = x_train[y_train == 1]    # 정답이 1인 데이터

# 결정 경계를 계산하기 위한 X, Y 좌표
xl = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)  # X 범위 생성
w1 = net.l1.weight.data[0, 0].item()  # 첫 번째 가중치
w2 = net.l1.weight.data[0, 1].item()  # 두 번째 가중치
b = net.l1.bias.data[0].item()        # 바이어스

# Y 좌표 계산
yl = -(b + w1 * xl) / w2

plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='b', label='0 (Setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='k', label='1 (Versicolor)')
plt.plot(xl, yl, c='b')
plt.xlabel('sepal_length')
plt.ylabel('sepla_width')
plt.legend()
plt.show()
