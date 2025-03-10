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

# ✅ 데이터셋 존재 여부 확인 후 다운로드
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

# ✅ ResNet50 모델 로드 (pretrained → weights 방식 변경)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 (100개 클래스)
model = model.to(device)

# ✅ 손실 함수 및 최적화 기법
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 모델 학습
num_epochs = 5  # 원하는 epoch 수 설정
log_interval = len(trainloader) // 5  # 5번 정도 loss 출력하도록 step 설정

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Accuracy 계산
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 일정 step마다 loss 출력
        if (batch_idx + 1) % log_interval == 0:
            print(f"[Epoch {epoch+1}, Step {batch_idx+1}/{len(trainloader)}] Loss: {running_loss / (batch_idx+1):.4f}")

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total

    # ✅ Test Loss & Accuracy 계산
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_loss /= len(testloader)
    test_acc = 100. * test_correct / test_total

    # ✅ 최종 결과 출력
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

print("Training Finished!")

# ✅ 모델 저장
torch.save(model.state_dict(), "resnet50_cifar100.pth")
print("Model saved!")
