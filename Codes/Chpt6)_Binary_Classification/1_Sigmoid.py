import numpy as np
import pandas as pd  # Pandas 라이브러리 추가
import matplotlib.pyplot as plt
import torch


x_np = np.arange(-4, 4.1, 0.25)

# x와 y 설정
x = torch.tensor(x_np).float()
y = torch.sigmoid(x)

# 그래프 출력
plt.title("Sigmoid Function")
plt.plot(x.data, y.data)
plt.show()