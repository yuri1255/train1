import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights
import os

torch.backends.cudnn.benchmark = True

# ✅ CIFAR-100 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 입력 크기
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ✅ 데이터셋 저장 경로 설정
data_path = "./data"

# ✅ 데이터셋이 존재하는지 확인 후 다운로드 방지
if not os.path.exists(os.path.join(data_path, "cifar-100-python")):
    download_flag = True
else:
    download_flag = False

# ✅ CIFAR-100 데이터셋 로드
batch_size = 32
trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=download_flag, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=download_flag, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# ✅ ResNet50 모델 불러오기 (pretrained → weights 방식 변경)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # 최신 방식으로 변경

# ✅ CIFAR-100은 클래스가 100개이므로 fc 레이어 변경
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# ✅ 손실 함수 및 최적화 기법
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 모델 학습
num_epochs = 5  # 원하는 epoch 수 설정
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

print("Training Finished!")

# ✅ 모델 저장
torch.save(model.state_dict(), "resnet50_cifar100.pth")
print("Model saved!")
