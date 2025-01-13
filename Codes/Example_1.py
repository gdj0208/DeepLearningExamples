import numpy as np
import matplotlib.pyplot as plt
import torch

sampleData = np.array([[166, 58.7],
                       [176.0, 75.7],
                       [171, 62.1],
                       [173, 70.4],
                       [169, 60.1]])

print(sampleData)

# 신장을 x로, 체중을 y로 설정
x = sampleData[:, 0]
y = sampleData[:, 1]

# 정규화
x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()

x = (x - x_mean) / x_std
y = (y - y_mean) / y_std

# X와 Y를 텐서 변수로 변환
x = torch.tensor(x).float()
y = torch.tensor(y).float()

# 초기화
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# 예측 함수 정의
def pred(X):
    return W * X + B

# 평균 제곱 오차 손실 함수 정의
def mse(Yp, Y):
    loss = ((Yp - Y) ** 2).mean()
    return loss

# 반복 횟수
num_epochs = 500

# 학습률
lr = 0.01

# 기록 저장용 배열
history = np.zeros((0, 2))

# 루프 처리
for epoch in range(num_epochs):
    # 예측 계산
    Yp = pred(x)

    # 손실 계산
    loss = mse(Yp, y)

    # 경사 계산
    loss.backward()

    with torch.no_grad():
        # 파라미터 수정
        W -= lr * W.grad
        B -= lr * B.grad

        # 경사값 초기화
        W.grad = None
        B.grad = None

    # 기록 저장 및 출력
    if epoch % 10 == 0:
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch} loss = {loss:.4f}')

# 최종 파라미터 값
print('W = ', W.data.numpy())
print('B = ', B.data.numpy())

# 손실 확인
print(f'초기상태 : 손실 : {history[0, 1]:.4f}')
print(f'최종상태 : 손실 : {history[-1, 1]:.4f}')

# 회귀선 시각화를 위한 X_range와 Y_range 계산
X_range = torch.linspace(min(x), max(x), 100)  # X 범위
Y_range = pred(X_range)  # 모델로 예측한 Y 값

plt.scatter(x, y, c='k', s=50)
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.plot(X_range.data, Y_range.data, lw=2, c='b')
plt.title("Relationhsip between Height & Weight")
plt.show()