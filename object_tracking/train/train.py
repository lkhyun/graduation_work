import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from PIL import Image
import os

os.chdir("/home/user2/radar_project")
log_file_path = f"./vit_training_log.txt"

def load_data(file_path):
    full_data = np.load(file_path)
    data = full_data[:, :-1].reshape(-1, 1, 72, 51)  # Reshape
    labels = full_data[:, -1]

    return data, labels

train_data_com, train_labels_com = [], []
test_data_com, test_labels_com = [], []
train_data_tv, train_labels_tv = [], []
test_data_tv, test_labels_tv = [], []

for i in range(1, 49):  # 1부터 40까지는 훈련 데이터
    data_com, labels_com = load_data(f"{i}_com.npy")
    data_tv, labels_tv = load_data(f"{i}_tv.npy")
    if i%6==0:
        test_data_com.append(data_com)
        test_labels_com.append(labels_com)
        test_data_tv.append(data_tv)
        test_labels_tv.append(labels_tv)
    else:
        train_data_com.append(data_com)
        train_labels_com.append(labels_com)
        train_data_tv.append(data_tv)
        train_labels_tv.append(labels_tv)

# Convert to tensors
train_data_com = torch.tensor(np.concatenate(train_data_com), dtype=torch.float32)
train_labels_com = torch.tensor(np.concatenate(train_labels_com), dtype=torch.int64)
train_data_tv = torch.tensor(np.concatenate(train_data_tv), dtype=torch.float32)
train_labels_tv = torch.tensor(np.concatenate(train_labels_tv), dtype=torch.int64)

test_data_com = torch.tensor(np.concatenate(test_data_com), dtype=torch.float32)
test_labels_com = torch.tensor(np.concatenate(test_labels_com), dtype=torch.int64)
test_data_tv = torch.tensor(np.concatenate(test_data_tv), dtype=torch.float32)
test_labels_tv = torch.tensor(np.concatenate(test_labels_tv), dtype=torch.int64)

class TransformTensorDataset(Dataset):
    def __init__(self, tensor_data, tensor_labels, transform=None):
        self.tensor_data = tensor_data
        self.tensor_labels = tensor_labels
        self.transform = transform

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        sample = self.tensor_data[idx].numpy().squeeze()  # Squeeze to remove the singleton dimension
        label = self.tensor_labels[idx]

        if self.transform:
            sample = Image.fromarray((sample * 255).astype(np.uint8))  # Convert to PIL Image
            sample = sample.convert('RGB')  # Convert to RGB
            sample = self.transform(sample)

        return sample, label

# Define the image transformations
image_size = 224  # Size expected by the ViT model
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# DataLoader 준비
train_dataset_com = TransformTensorDataset(train_data_com, train_labels_com, transform=transform)
train_loader_com = DataLoader(train_dataset_com, batch_size=16, shuffle=True)
train_dataset_tv = TransformTensorDataset(train_data_tv, train_labels_tv, transform=transform)
train_loader_tv = DataLoader(train_dataset_tv, batch_size=16, shuffle=True)

test_dataset_com = TransformTensorDataset(test_data_com, test_labels_com, transform=transform)
test_loader_com = DataLoader(test_dataset_com, batch_size=16, shuffle=False)
test_dataset_tv = TransformTensorDataset(test_data_tv, test_labels_tv, transform=transform)
test_loader_tv = DataLoader(test_dataset_tv, batch_size=16, shuffle=False)

# 모델 학습 및 평가
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_com = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=9).to(device)
criterion_com = nn.CrossEntropyLoss()
optimizer_com = optim.Adam(model_com.parameters(), lr=1e-4)
scheduler_com = lr_scheduler.StepLR(optimizer_com, step_size=7, gamma=0.1)

model_tv = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=7).to(device)
criterion_tv = nn.CrossEntropyLoss()
optimizer_tv = optim.Adam(model_tv.parameters(), lr=1e-4)
scheduler_tv = lr_scheduler.StepLR(optimizer_tv, step_size=7, gamma=0.1)

def train_and_evaluate(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=10):


    with open(log_file_path, "a") as log_file:
        log_file.write("Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # Epoch 결과 출력
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # 로그 파일에 기록
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{epoch + 1},{epoch_loss:.4f},{epoch_acc:.4f}\n")

def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # 모델을 평가 모드로 전환
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / test_total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
    return test_loss, test_acc

# 모델 학습
train_and_evaluate(model_com, train_loader_com, criterion_com, optimizer_com, scheduler_com, device, num_epochs=10)
torch.save(model_com.state_dict(), './model_com.pth')
train_and_evaluate(model_tv, train_loader_tv, criterion_tv, optimizer_tv, scheduler_tv, device, num_epochs=10)
torch.save(model_tv.state_dict(), './model_tv.pth')
# 테스트 데이터로 모델 평가
evaluate_model(model_com, test_loader_com, criterion_com, device)
evaluate_model(model_tv, test_loader_tv, criterion_tv, device)
