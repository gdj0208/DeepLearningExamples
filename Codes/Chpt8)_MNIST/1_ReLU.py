import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


'''
# < 1. ReLU 함수 > ===========================================
# ReLU 함수를 보여주기 위한 코드
relu = nn.ReLU()

# x, y 설정
x_np = np.arange(-2, 2.1, 0.25)
x = torch.tensor(x_np).float()
y = relu(x)

# ReLU 함수 출력
plt.plot(x.data, y.data)
plt.show()


# < 2. GPU 사용법 > ==========================================
# 디바이스 할당 (Window OS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# GPU 사용 가능 시 cuda:0 출력
# 사용 불가시 cpu 출력

# 디바이스 할당 (MacOS)
print(torch.backends.mps.is_available())  # True면 MPS 사용 가능
print(torch.backends.mps.is_built())      # PyTorch가 MPS를 지원하는지 확인
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# CPU->GPU로 데이터 보내기
x_np = np.arange(-2, 2.1, 0.25)
y_np = np.arange(-2, 2.1, 0.25)
x = torch.tensor(x_np).float()
y = torch.tensor(y_np).float()
x = x.to(device)

print(x.device)
print(y.device)

# to()를 활용해서 모델을 GPU로 보낼 수 도 있음
# net = Net(n_input, n_output, n_hidden).to(device) 
'''


