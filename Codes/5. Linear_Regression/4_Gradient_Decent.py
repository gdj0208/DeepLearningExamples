
import numpy as np
import pandas as pd  # Pandas 라이브러리 추가
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# < 0. 모델 정의 > ================================================================
class Net(nn.Module) :
    def __init__(self, n_input, n_output) :

        # 부모 클래스 nn.Module의 초기화 호출
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(n_input, n_output)

        #초기값을 모두 1롤 설정
        nn.init.constant_(self.l1.weight, 0.0)
        nn.init.constant_(self.l1.bias, 0.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1
    

# < 1. 입력값 x와 텐서 yt의 텐서변환 > ================================================
# 데이터 URL
file_loc = "Codes/5. Linear_Regression/"
file_path = "BostonHousing.csv"

# 데이터 로드
boston_df = pd.read_csv(file_loc+file_path)

features_name = np.array(['crim', 'zn', 'indus', 'chas', 'nox' ,'rm' ,'age' ,'rad' ,'tax' ,'ptratio', 'b', 'lstat', 'medv'])

x = boston_df['rm']  # [[값], [값], ...] 형태로 변환
yt = boston_df['medv']


# 입력값과 정답의 텐서변환
inputs = torch.tensor(x).float()
labels = torch.tensor(yt).float() 
# labels.shpae == torch.Size([5])

# N차원 벡터에서 (N,1)행렬로 전환
inputs = inputs.view((-1, 1))
labels = labels.view((-1, 1))
# labels.shape == torch.Size([5,1])



# < 2. 초기화 설정 > ==========================================

# 학습률 정의
lr = 0.001

# 입력과 출력 차원을 명시
net = Net(1, 1)                    

# 손실함수 정의
criterion = nn.MSELoss()

# 최적화함수 경사하강으로 정의
optimizer = optim.SGD(net.parameters(), lr = lr)

# 평가 결과 기록 (손실 값만을 기록)
history = np.zeros((0,2))

# 이전 손실 값
prev_loss = 0

# 충분한 학습 여부의 판별기
pass_discriminator = 0.000001


# < 3. 경사하강 학습 > =============================================
epoch = 0

while(True) :
    # 경사값 초기화
    # 반복 처리마다 경사값을 초기화하지 않을 시 경사 값은 경사 계산이 될 대 마다 계속 더해진다.
    # 따라서 파라미터 수정이 끝나면 경사값을 초기화 해야 한다.      
    optimizer.zero_grad()


    # 예측 계산 
    # inputs도 (N, 1) 형태로 변환해야 합니다.
    outputs = net(inputs) 


    # 손실 계산 
    loss = criterion(labels, outputs)
    print(f'{loss.item() : .5f}')


    # 경사 계산
    loss.backward()

    # 경사 계산 결과를 취득 가능하도록 함
    #print(net.l1.weight.grad)
    #print(net.l1.bias.grad)


    # 파라미터 수정
    optimizer.step()


    # 100회마나다 경과 기록
    if(epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss : {loss.item(): .5f}')


    # 충분한 학습이 되었는지 판별
    # 이전 손실 값과의 차이가 0.000001 이하면 충분히 모델링 된 것으로 취급
    if(abs(loss - prev_loss) <= pass_discriminator) :
        break

    # 충분한 학습이 되지 않았을 시 저번 학습의 손실값을 저장
    prev_loss = loss
    epoch += 1


# < 4. 결과 확인 > ==================================================



# 산포도 & 회귀직선
plt.scatter(x, yt, s=5, c='b')
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.plot(x, outputs.detach().numpy(), c='k')
plt.title('Relationship between Rooms and Price in Boston')
plt.show()
