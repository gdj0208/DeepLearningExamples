
import numpy as np

# 라이브러리 임포트
import torch

# < 1. 다양한 계수의 텐서 > ========================================

# 0계 텐서
r0 = torch.tensor(1.0).float()
print(type(r0)) # type 출력  : <class 'torch.Tensor'>
print(r0.dtype) # dtype 출력 : torch.float32
print(r0.shape) # shape 출력 : torch.Size([])
print(r0.data)  # 데이터 출력  : tensor(1.)



# 1계 텐서
r1_np =  np.array([1, 2, 3, 4, 5]) # 1계 텐서(벡터)
print(r1_np.shape) # (5, )

# 넘파이에서 텐서로 변환
r1 = torch.tensor(r1_np).float()
print(r1.dtype) # dtype 출력 : torch.float32
print(r1.shape) # shape 출력 : torch.Size([5])
print(r1.data)  # 데이터 출력  : tensor([1., 2., 3., 4., 5.])



# 2계 텐서
r2_np =  np.array( [[1, 2, 3], [4, 5, 6]] ) # 2계 텐서(행렬)
print(r2_np.shape) # (2, 3)

# 넘파이에서 텐서로 변환
r2 = torch.tensor(r2_np).float()
print(r2.shape) # shape 출력 : torch.Size([2, 3])
print(r2.data)  # 데이터 출력  : tensor( [[1., 2., 3.], [4., 5., 6.]] )


# 3계 텐서
# shape = [3,2,2]의 정규분포 텐서 작성
r3 = torch.randn((3,2,2))
print(r3.shape) # shape 출력 : torch.Size([3, 2, 2])
print(r3.data)  # 데이터 출력  : tensor( [[1., 2.], ... ,[4., 5.]] )


# < 2. 텐서 함수 > ========================================
# view() : 텐서 계수 변환
r4 = torch.randn((3,2,2))   # shape = [3,2,2]의 정규분포 텐서 작성
print(r4.shape)

r5 = r4.view(3, -1)         # 요소 수를 -1로 지정시 이 수를 자동 조정함
print(r5.shape)

# item() : 텐서의 요소 하나의 데이터값 출력
r1 = np.array([1, 2, 3, 4])
print(r1.item()) # 오류 발생 (하나를 지정해야 한다.)

r2 = torch.ones(1)
print(r2.item()) # 출력값 : 1.0
