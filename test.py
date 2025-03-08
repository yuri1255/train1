import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

torch.backends.cudnn.benchmark = True

# ✅ CIFAR-100 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 입력 크기
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ✅ CIFAR-100 데이터셋 로드
batch_size = 32
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform, )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# ✅ ResNet50 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)

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
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

print("Training Finished!")

# ✅ 모델 저장
torch.save(model.state_dict(), "resnet50_cifar100.pth")
print("Model saved!")
