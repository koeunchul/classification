import torch
import torch.nn as nn
from torch import optim
from torchvision import utils
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchsummary import summary
import os
import numpy as np
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

# 채널 별 mean 계산
def get_mean(dataset):
    meanRGB = [np.mean(image.numpy(), axis=(1,2)) for image,_ in dataset]
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    return [meanR, meanG, meanB]

# 채널 별 str 계산
def get_std(dataset):
    stdRGB = [np.std(image.numpy(), axis=(1,2)) for image,_ in dataset]
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    return [stdR, stdG, stdB]

def train():
    train_dataset   = datasets.ImageFolder(root='./dataset_classification/train', transform=transforms.ToTensor())
    test_dataset    = datasets.ImageFolder(root='./dataset_classification/test' , transform=transforms.ToTensor())

    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(get_mean(train_dataset), get_std(train_dataset))])
    test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(get_mean(test_dataset), get_std(test_dataset))])

    # trainsform 정의
    train_dataset.transform = train_transforms
    test_dataset.transform = test_transforms

    # dataloader 정의
    train_dataloader    = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader     = DataLoader(test_dataset, batch_size=64, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 학습 환경 설정
    model = models.efficientnet_b5(pretrained=True).to(device) # true 옵션으로 사전 학습된 모델을 로드
    # model = models.resnet50(pretrained=True).to(device) # true 옵션으로 사전 학습된 모델을 로드

    summary(model, (3, 128, 128))

    lr = 0.0001
    num_epochs = 1000
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss().to(device)

    params = {
        'num_epochs':num_epochs,
        'optimizer':optimizer,
        'loss_function':loss_function,
        'train_dataloader':train_dataloader,
        'test_dataloader': test_dataloader,
        'device':device
    }
    for epoch in range(0, num_epochs):
      for i, data in enumerate(train_dataloader, 0):
        # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 이전 batch에서 계산된 가중치를 초기화
        optimizer.zero_grad() 

        # forward + back propagation 연산
        outputs = model(inputs)
        train_loss = loss_function(outputs, labels)
        train_loss.backward()
        optimizer.step()

      # test accuracy 계산
      total = 0
      correct = 0
      accuracy = []
      for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 결과값 연산
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss = loss_function(outputs, labels).item()
        accuracy.append(100 * correct/total)

      # 학습 결과 출력
      print('Epoch: %d/%d, Train loss: %.6f, Test loss: %.6f, Accuracy: %.2f' %(epoch+1, num_epochs, train_loss.item(), test_loss, 100*correct/total))


def resize():
    resize_size = 256
    dir_path = './dataset_classification/test/crack'

    # 이미지 전처리를 위한 transform 정의
    resize_transform = transforms.Compose([transforms.Resize((resize_size, resize_size))])
    files = [f for f in os.listdir(dir_path)]
    for file in files:
        # 이미지를 로드
        img_path = os.path.join(dir_path, file)
        img = Image.open(img_path)
        
        # 이미지를 256x256 크기로 resize
        img_resized = resize_transform(img)
        
        # 리사이즈된 이미지 저장
        save_path = os.path.join(dir_path, file)
        img_resized.save(save_path)
    print("All images have been resized to 256x256!")


if __name__ == '__main__':
    train()
    # resize()