import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os

mapping_com = {0:0,1:1,2:2,3:3,4:4,5:10,6:11,7:12,8:13}
mapping_tv = {0:8,1:9,2:10,3:4,4:5,5:6,6:7}
log_file_path = f"./vit_training_log.txt"

class CombinedTensorDataset(Dataset):
    def __init__(self, tensor_data_com, tensor_data_tv, tensor_labels, transform=None):
        self.tensor_data_com = tensor_data_com
        self.tensor_data_tv = tensor_data_tv
        self.tensor_labels = tensor_labels
        self.transform = transform

    def __len__(self):
        return len(self.tensor_labels)

    def __getitem__(self, idx):
        # 각각의 데이터 추출
        sample_com = self.tensor_data_com[idx].numpy().squeeze()
        sample_tv = self.tensor_data_tv[idx].numpy().squeeze()
        label = self.tensor_labels[idx]

        # 필요 시 변환 적용
        if self.transform:
            sample_com = Image.fromarray((sample_com * 255).astype(np.uint8))  # com 데이터를 PIL로 변환
            sample_com = sample_com.convert('RGB')  # RGB로 변환
            sample_com = self.transform(sample_com)  # 변환 적용

            sample_tv = Image.fromarray((sample_tv * 255).astype(np.uint8))  # tv 데이터를 PIL로 변환
            sample_tv = sample_tv.convert('RGB')  # RGB로 변환
            sample_tv = self.transform(sample_tv)  # 변환 적용

        return (sample_com, sample_tv), label  # 두 데이터를 함께 반환

# 이미지 변환 정의 (ViT 등에서 사용하는 형태로 변환)
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def predict_model(model1, model2, dataset, device):
    model1.eval()  # 모델을 평가 모드로 전환
    model2.eval()
    test_total = 0
    test_correct = 0
    test_acc = 0.0
    prev_value_com = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for (input_com, input_tv), labels in dataset:
            input_com, input_tv, labels = input_com.to(device), input_tv.to(device), labels.to(device)
            output_com = model1(input_com)
            output_tv = model2(input_tv)
            _, predicted_com = torch.max(output_com, 1)
            _, predicted_tv = torch.max(output_tv, 1)
            if predicted_com.item() == 4:
                if prev_value_com == 4:
                    predicted = mapping_tv[predicted_tv.item()]
                else:
                    predicted = mapping_com[predicted_com.item()]

            else:
                predicted = mapping_com[predicted_com.item()]

            prev_value_com = predicted_com.item()

            all_predictions.append(predicted)
            all_labels.append(labels.item())

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

        test_acc = test_correct / test_total

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",cbar=False, xticklabels=range(14),
                yticklabels=range(14))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix")
    plt.show()
    print(f"predict:{test_acc}\n")
    return

def run(full_data):
    os.chdir(f"C:/Users/dlrkd/Desktop/graduation_work/object_tracking")
    labels_all = full_data[:, -1].copy()
    full_data = full_data[:,:-1]
    data_com = full_data[:, :full_data.shape[1]//2].reshape(-1, 1, 72, 51)  # Reshape
    data_tv = full_data[:, full_data.shape[1]//2: ].reshape(-1, 1, 72, 51)  # Reshape

    test_data_com = torch.tensor(data_com, dtype=torch.float32)
    test_data_tv = torch.tensor(data_tv, dtype=torch.float32)
    test_labels_all = torch.tensor(labels_all, dtype=torch.int64)

    # 데이터셋 정의
    combined_dataset = CombinedTensorDataset(test_data_com, test_data_tv, test_labels_all, transform=transform)
    test_dataloader = DataLoader(combined_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_com = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=9).to(device)
    model_com.load_state_dict(torch.load('./model_com.pth',weights_only=True))
    model_tv = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=7).to(device)
    model_tv.load_state_dict(torch.load('./model_tv.pth',weights_only=True))
    predict_model(model_com,model_tv,test_dataloader,device)

if __name__ == "__main__":
    full_data = np.load(f"finaltest.npy")
    run(full_data)