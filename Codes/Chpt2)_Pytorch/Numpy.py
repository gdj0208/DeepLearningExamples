
# 라이브러리 임포트
import numpy as np

# <1. 정의 > ========================================

# array 함수를 활용한 정의

# array 함수를 통한 벡터(1계 배열) 변수 정의
n1 = np.array([1,2,3,4,5])
print(n1)                   # 결과 확인
print(n1.shape)             # 요소 수 확인 1
print(len(n1))              # 요소 수 확인 2

# array 함수를 통한 행렬(2계 배열) 변수 정의
n2 = np.array([[1,2,3],[4,5,6]])
print(n2)
print(n2.shape)             # (3,2)
print(len(n2))              # 4

# zero 함수를 활용한 정의
n3 = np.zeros((3,2))
print(n3)

# ones 함수를 활용한 정의
n4 = np.zeros((3,4,5))
print(n4)

# random.randn 함수를 활용한 정의
n5 = np.random.randn(3,2,2)
print(n5)


# < 4. 그래프 출력용 수치 배열 생성 > ================================
# linspace를 활용한 정의
n6 = np.linspace(-1, 1, 11)
print(n6)                   
# 출력 결과 : [-1.  -0.8 -0.6 -0.4 -0.2  0.   0.2  0.4  0.6  0.8  1. ]

# arange를 활용한 정의
n7 = np.arange(-1, 1.2, 0.2)
print(n7)
# 출력 결과 : [-1, -0.8, -0.6, ..... , 0.8, 1.0 ]


# < 5. 조작 > =====================================

# 특정 행, 렬 추출
print(n2)
print(n2[:, 2]) # 모든 행의 2번 열들을 출력

# 행렬, 벡터 형태 변경
n8 = np.array(range(24))
print(n8)
n8 = n8.reshape(4,6)
print(n8)
n8 = n8.reshape(3, -1)  # 열의 인자로 -1이 입력된 결과로 열의 수가 자동으로 계산되어짐
print(n8)

# 전치행렬
print(n2)
n9 = n2.T
print(n9)

# 행렬 붙이기
n10 = np.array(range(1,7)).reshape(2,3)
n11 = np.array(range(7,13)).reshape(2,3)
print(n10)
print(n11)

# 세로 연결하기
n12 = np.vstack([n10, n11])
print(n12)

# 가로 연결
n13 = np.hstack([n10, n11])
print(n13)

