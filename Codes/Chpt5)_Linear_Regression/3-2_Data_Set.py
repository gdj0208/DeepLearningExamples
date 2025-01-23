
import numpy as np
import pandas as pd  # Pandas 라이브러리 추가
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

# < 1. 학습용 데이터 준비 > ===================================================

# '보스턴 데이터셋'은 현재 사이킷런 라이브러리에서 가져올 수 있지만,
# 사이킷런에서 이 데이터를 앞으로 사용할 수 없기에 csv파일로 다운받아서 실행했습니다.

# 데이터 URL
file_loc = "Codes/5. Linear_Regression/"
file_path = "BostonHousing.csv"

# 데이터 로드
boston_df = pd.read_csv(file_loc+file_path)

features_name = np.array(['crim', 'zn', 'indus', 'chas', 'nox' ,'rm' ,'age' ,'rad' ,'tax' ,'ptratio', 'b', 'lstat', 'medv'])

# 데이터 확인
print(boston_df.head())  # 상위 5개 행 출력
print("데이터 크기: {boston_df.shape}")  # 데이터 크기 출력


# < 2. 두 데이터간 산포도 구하기 > =====================================================
# 방 개수와 가격의 산포도를 구하기

# 방의 개수 추출
rm = boston_df['rm']  # [[값], [값], ...] 형태로 변환
medv = boston_df['medv']

# 결과 출력 ( 상위 5개의 방 정보만을 출려)
plt.scatter(rm, medv, s=10, c="b")
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.title('Relationship between RM, MEDV')
plt.show()