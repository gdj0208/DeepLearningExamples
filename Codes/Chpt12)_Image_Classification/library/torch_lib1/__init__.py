# 공통 함수 정의
# Apache License Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0.html

# ReadMe
README = 'Common Library for PyTorch\nAuthor: M. Akaishi'

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.notebook import tqdm

# 손실 함수 계산
def eval_loss(loader, device, net, criterion):
  
    # 데이터로더에서 처음 한 세트 가져오기
    for images, labels in loader:
        break

    # 디바이스 할당
    inputs = images.to(device)
    labels = labels.to(device)

    # 예측 계산
    outputs = net(inputs)

    # 손실 계산
    loss = criterion(outputs, labels)

    return loss
  

# 학습용 함수
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):

    base_epochs = len(history)
  
    for epoch in range(base_epochs, num_epochs+base_epochs):
        print(f'{epoch} / {base_epochs + num_epochs}')
        
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # 훈련 페이즈
        net.train()
        count = 0

        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 경사 초기화
            optimizer.zero_grad()

            # 예측 함수
            outputs = net(inputs)

            # 손실 함수
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 경사 계산
            loss.backward()

            # 파라미터 수정
            optimizer.step()

            # 예측 라벨 산출
            predicted = torch.max(outputs, 1)[1]

            # 정답 건수 산출
            train_acc += (predicted == labels).sum().item()

            # 손실과 정확도 계산
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        # 예측 페이즈
        net.eval()
        count = 0

        for inputs, labels in test_loader:
            count += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 예측 계산
            outputs = net(inputs)

            # 손실 계산
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 예측 라벨 산출
            predicted = torch.max(outputs, 1)[1]

            # 정답 건수 산출
            val_acc += (predicted == labels).sum().item()

            # 손실과 정확도 계산
            avg_val_loss = val_loss / count
            avg_val_acc = val_acc / count
    
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history



# 학습 로그 해석
def evaluate_history(history):
  # 손실과 정확도 확인
  print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}')
  print(f'최종상태 : 손실 : {history[-1,3]:.5f} 정확도 : {history[-1,4]:.5f}' )

  num_epochs = len(history)
  if num_epochs < 10:
    unit = 1
  else:
    unit = num_epochs / 10

  # 학습 곡선 출력(손실)
  plt.figure(figsize=(9,8))
  plt.plot(history[:,0], history[:,1], 'b', label='Train')
  plt.plot(history[:,0], history[:,3], 'k', label='Test')
  plt.xticks(np.arange(0,num_epochs+1, unit))
  plt.xlabel('Repeat')
  plt.ylabel('Looss')
  plt.title('Learning Rate(Loss)')
  plt.legend()
  plt.show()

  # 학습 곡선 출력(정확도)
  plt.figure(figsize=(9,8))
  plt.plot(history[:,0], history[:,2], 'b', label='Train')
  plt.plot(history[:,0], history[:,4], 'k', label='Test')
  plt.xticks(np.arange(0,num_epochs+1,unit))
  plt.xlabel('Repeat')
  plt.ylabel('Accuacy')
  plt.title('Learning Rate(Accuracy)')
  plt.legend()
  plt.show()


# 이미지와 라벨 표시
def show_images_labels(loader, classes, net, device):

    # 데이터로더에서 처음 한 세트 가져오기
    for images, labels in loader:
        break
    # 표시 수는 50개
    n_size = min(len(images), 50)

    if net is not None:
      # 디바이스 할당
      inputs = images.to(device)
      labels = labels.to(device)

      # 예측 계산
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]
      #images = images.to('cpu')

    # 처음 n_size개 표시
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # net이 None이 아닌 경우는 예측 결과도 타이틀에 표시
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 정답 여부를 색으로 나타냄
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # net이 None인 경우는 정답 라벨만을 표시
        else:
          ax.set_title(label_name, fontsize=20)
        # 텐서를 넘파이 배열로 변환
        image_np = images[i].numpy().copy()
        # 축의 순서 변경(channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 값의 범위를[-1, 1] -> [0, 1]로 되돌림
        img = (img + 1)/2
        # 결과 출력
        plt.imshow(img)
        ax.set_axis_off()
    plt.show()


# 파이토치 난수 고정

def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
