import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from PIL import Image
import os

os.chdir("/home/user2/radar_project")
log_file_path = f"./vit_training_log.txt"

def load_data(file_path):
    full_data = np.load(file_path)
    data = full_data[:, :-1].reshape(-1, 1, 72, 51)  # Reshape
    labels = full_data[:, -1]
    
    # # Normalize data(z-score)
    # mean = data.mean(axis=0, keepdims=True)
    # std = data.std(axis=0, keepdims=True)
    # std[std == 0] = 1
    # data = (data - mean) / std
    
    return data, labels

# Data preparation
all_data_com, all_labels_com = [], []
all_data_tv, all_labels_tv = [], []

for i in range(1, 49):
    data_com, labels_com = load_data(f"{i}_com.npy")
    data_tv, labels_tv = load_data(f"{i}_tv.npy")
    all_data_com.append(data_com)
    all_labels_com.append(labels_com)
    all_data_tv.append(data_tv)
    all_labels_tv.append(labels_tv)

# Convert to tensors
all_data_com = torch.tensor(np.concatenate(all_data_com), dtype=torch.float32)
all_labels_com = torch.tensor(np.concatenate(all_labels_com), dtype=torch.int64)
all_data_tv = torch.tensor(np.concatenate(all_data_tv), dtype=torch.float32)
all_labels_tv = torch.tensor(np.concatenate(all_labels_tv), dtype=torch.int64)

# 전체 데이터에서 subject 단위로 test data를 분리
def split_data_by_subject(data, labels, groups, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(data, labels, groups=groups):
        train_data, test_data = data[train_idx], data[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        return train_data, test_data, train_labels, test_labels

# Create a wrapper for TensorDataset to apply transformations
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


# Groups for group k-fold
groups_com = np.concatenate([[i] * 84 for i in range(1, 49)])
groups_tv = np.concatenate([[i] * 84 for i in range(1, 49)])

# Define the image transformations
image_size = 224  # Size expected by the ViT model
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Subject 단위로 test data를 분리
train_data_com, test_data_com, train_labels_com, test_labels_com = split_data_by_subject(all_data_com, all_labels_com, groups_com, test_size=8)
train_data_tv, test_data_tv, train_labels_tv, test_labels_tv = split_data_by_subject(all_data_tv, all_labels_tv, groups_tv, test_size=8)

# Test 데이터셋을 위한 DataLoader 준비
test_dataset_com = TransformTensorDataset(test_data_com, test_labels_com, transform=transform)
test_loader_com = DataLoader(test_dataset_com, batch_size=84, shuffle=False)

test_dataset_tv = TransformTensorDataset(test_data_tv, test_labels_tv, transform=transform)
test_loader_tv = DataLoader(test_dataset_tv, batch_size=84, shuffle=False)


def train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=30, model_save_prefix="best_model",fold=0):
    best_val_acc = 0.0
    best_epoch = 0
    model_save_path = ''

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

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(valid_loader.dataset)
        val_acc = val_correct / val_total

        # Epoch 결과 출력
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # 로그 파일에 기록
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{epoch + 1},{epoch_loss:.4f},{epoch_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        # Check if this epoch's validation accuracy is the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # Save model with epoch and accuracy in the filename
            model_save_path = f"./{model_save_prefix}_fold{fold}.pth"
            torch.save(model.state_dict(), model_save_path)

    # 로그 파일에 기록
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Model saved as {model_save_path}\n")
        log_file.write(f"Best Validation Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}\n")

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
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Accuracy: {((predicted == labels).sum().item())/labels.size(0)}\n")


    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / test_total

    with open(log_file_path, "a") as log_file:
        log_file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
    return test_loss, test_acc

# 저장된 모델 불러오기 및 test 데이터로 평가
def load_and_evaluate(model_save_path, test_loader, num_classes, criterion, device, base_model):
    model = timm.create_model(base_model, pretrained=False, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_save_path))  # 저장된 모델 가중치 불러오기
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Model: {base_model}\n")

    evaluate_model(model, test_loader, criterion, device)

groups_com = np.concatenate([[i] * 14 for i in range(1, 241)])
groups_tv = np.concatenate([[i] * 14 for i in range(1, 241)])

# GroupKFold cross-validation
group_kfold = GroupKFold(n_splits=5)
model_lists = ['vit_small_patch16_224']
for model_list in model_lists:
    # Cross-validation for COM data
    for fold, (train_idx, valid_idx) in enumerate(group_kfold.split(train_data_com, train_labels_com, groups=groups_com)):
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Fold {fold + 1}\n")

        # Prepare models, criteria, optimizers, and schedulers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_com = timm.create_model(model_list, pretrained=True, num_classes=9).to(device)
        criterion_com = nn.CrossEntropyLoss()
        optimizer_com = optim.Adam(model_com.parameters(), lr=1e-4)
        scheduler_com = lr_scheduler.StepLR(optimizer_com, step_size=7, gamma=0.1)

        train_dataset_com = TransformTensorDataset(train_data_com[train_idx], train_labels_com[train_idx], transform=transform)
        valid_dataset_com = TransformTensorDataset(train_data_com[valid_idx], train_labels_com[valid_idx], transform=transform)

        train_loader_com = DataLoader(train_dataset_com, batch_size=16, shuffle=True)
        valid_loader_com = DataLoader(valid_dataset_com, batch_size=16, shuffle=False)

        train_and_evaluate(model_com, train_loader_com, valid_loader_com, criterion_com, optimizer_com, scheduler_com, device, model_save_prefix=f"{model_list}_com",fold=fold+1)

    # Cross-validation for TV data
    for fold, (train_idx, valid_idx) in enumerate(group_kfold.split(train_data_tv, train_labels_tv, groups=groups_tv)):
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Fold {fold + 1}\n")

        # Prepare models, criteria, optimizers, and schedulers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_tv = timm.create_model(model_list, pretrained=True, num_classes=7).to(device)
        criterion_tv = nn.CrossEntropyLoss()
        optimizer_tv = optim.Adam(model_tv.parameters(), lr=1e-4)
        scheduler_tv = lr_scheduler.StepLR(optimizer_tv, step_size=7, gamma=0.1)

        train_dataset_tv = TransformTensorDataset(train_data_tv[train_idx], train_labels_tv[train_idx], transform=transform)
        valid_dataset_tv = TransformTensorDataset(train_data_tv[valid_idx], train_labels_tv[valid_idx], transform=transform)

        train_loader_tv = DataLoader(train_dataset_tv, batch_size=16, shuffle=True)
        valid_loader_tv = DataLoader(valid_dataset_tv, batch_size=16, shuffle=False)

        train_and_evaluate(model_tv, train_loader_tv, valid_loader_tv, criterion_tv, optimizer_tv, scheduler_tv, device, model_save_prefix=f"{model_list}_tv",fold=fold+1)

for model_list in model_lists:
    for i in range(1,6):
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{model_list}_fold{i}_com result\n")

        # COM 데이터 평가
        model_save_path_com = f"./{model_list}_com_fold{i}.pth"  # 저장된 모델 경로
        criterion_com = nn.CrossEntropyLoss()
        load_and_evaluate(model_save_path_com, test_loader_com, num_classes=9, criterion=criterion_com, device=device, base_model=model_list)

        with open(log_file_path, "a") as log_file:
            log_file.write(f"{model_list}_fold{i}_tv result\n")

        # TV 데이터 평가
        model_save_path_tv = f"./{model_list}_tv_fold{i}.pth"  # 저장된 모델 경로
        criterion_tv = nn.CrossEntropyLoss()
        load_and_evaluate(model_save_path_tv, test_loader_tv, num_classes=7, criterion=criterion_tv, device=device, base_model=model_list)
