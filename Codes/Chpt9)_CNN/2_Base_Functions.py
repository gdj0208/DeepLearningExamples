
import torch
import numpy as np

# < 1. 손실 계산 > ===============================================
# 손실 계산
def eval_loss(loader, device, net, criterion) :
    # 데이터 로더에서 한 세트 가져오기
    for images, labels in loader:
        break

    # 디바이스 할당
    inputs = images.to(device)
    outpus = labels.to(device)

    outputs = net(inputs)   # 예측 계산
    loss = criterion(outputs, labels)

    return loss


# < 2. 학습 > ===============================================
# 학습
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
    from tqdm.notebook import tqdm

    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs+base_epochs):
        train_loss, train_acc = 0
        val_loss, val_acc = 0
        n_train, n_test = 0

        # 훈련 페이즈
        net.train()

        for inputs, labels in tqdm(train_loader):
            n_train += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 경사 초기화
            optimizer.zero_grad()
            
            outputs = net(inputs)               # 예측 계산
            loss = criterion(outputs, labels)   # 손실 계산
            loss.backward()                     # 경사 계산
            optimizer.step()                    # 파라미터 수정
            
            predicted = torch.max(outputs, 1)[1]  # 예측 라벨 산출

            # 손실과 정확도 계산
            train_loss += loss.item()
            train_acc += (predicted==labels).sum().item()


        # 예측 페이즈
        net.eval()
        count = 0

        for inputs, labels in test_loader:
            n_test += len(labels)
            inputs = inputs.to(device)
            lables = lables.to(device)

            outputs_test = net(inputs)                 # 예측 계산
            loss_test = criterion(outputs_test, lables)# 손실 계산
            predicted_test = torch.max(outputs_test, 1)[1]  # 예측 라벨 산출

            # 손실과 정확도 계산
            val_loss += loss_test.item()
            val_acc += (predicted_test==lables).sum().item()


        train_acc /= n_train
        val_acc /= n_test

        if((epoch)% 10 == 0) :
            item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
            history = np.vstack((history, item))