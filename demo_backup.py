import os
# 解决OpenMP库重复初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
from tqdm import tqdm  # 导入tqdm库

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(42)


# 全局配置
class Config:
    DATA_DIR = r"D:\pythondata\机器学习实验\fish_classification\Data"
    OUTPUT_DIR = "./output"
    MODEL_DIR = "./models"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    NUM_CLASSES = 23
    IMG_SIZE = 300
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 类别名称
    CLASSES = [
        'Clams', 'Corals', 'Crabs', 'Dolphins', 'Eels', 'Fish',
        'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus', 'Penguin',
        'Puffers', 'Rays', 'Sea Otter', 'Sea Urchins', 'Seahorse',
        'Seal', 'Shrimp', 'Squid', 'Starfish', 'Whales'
    ]


# 创建输出目录
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)


# ==================== 1. 数据预处理 ====================

class SeaAnimalDataset(Dataset):
    """海洋动物数据集类"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')

            # 等比例缩放+填充到300x300
            image = self.resize_and_pad(image)

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个黑色图像
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label

    def resize_and_pad(self, image):
        """等比例缩放并填充到300x300"""
        w, h = image.size

        # 计算缩放比例
        if w > h:
            new_w = Config.IMG_SIZE
            new_h = int(h * Config.IMG_SIZE / w)
        else:
            new_h = Config.IMG_SIZE
            new_w = int(w * Config.IMG_SIZE / h)

        # 缩放
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # 创建黑色背景并粘贴图像
        new_image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))
        paste_x = (Config.IMG_SIZE - new_w) // 2
        paste_y = (Config.IMG_SIZE - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))

        return new_image


def load_dataset():
    """加载数据集并划分训练/验证/测试集"""
    print("Loading dataset...")

    image_paths = []
    labels = []
    class_to_idx = {}

    # 遍历数据目录
    for idx, class_name in enumerate(sorted(os.listdir(Config.DATA_DIR))):
        class_path = os.path.join(Config.DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        class_to_idx[class_name] = idx

        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(idx)

    print(f"Total images: {len(image_paths)}")
    print(f"Number of classes: {len(class_to_idx)}")

    # 统计类别分布
    class_counts = Counter(labels)
    print("\nClass distribution:")
    for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {class_name}: {class_counts[idx]}")

    # 分层划分数据集
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=0.15, stratify=labels, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )

    print(f"\nTrain set: {len(X_train)}")
    print(f"Val set: {len(X_val)}")
    print(f"Test set: {len(X_test)}")

    # 计算类别权重
    class_weights = compute_class_weights(y_train)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx, class_weights


def compute_class_weights(labels):
    """计算类别权重以处理不平衡"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)

    weights = torch.zeros(num_classes)
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)

    return weights


# ==================== 2. 数据增强策略 ====================

def get_transforms(augmentation_group='A'):
    """获取数据增强变换"""

    # 基础变换(验证集和测试集)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 训练集增强 - 组A
    train_transform_A = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 训练集增强 - 组B
    train_transform_B = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ColorJitter(contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if augmentation_group == 'A':
        return train_transform_A, base_transform
    else:
        return train_transform_B, base_transform


# ==================== 3. 基础模型定义 ====================

class BaseModel(nn.Module):
    """基础模型包装器"""

    def __init__(self, model_name, num_classes, pretrained=True):
        super(BaseModel, self).__init__()
        self.model_name = model_name

        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes)
            )
            # 冻结前10层
            ct = 0
            for child in self.model.children():
                ct += 1
                if ct < 7:  # ResNet的前几个block
                    for param in child.parameters():
                        param.requires_grad = False

        elif model_name == 'mobilenet_v3':
            self.model = models.mobilenet_v3_large(pretrained=pretrained)
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)
            # 冻结前5层
            ct = 0
            for child in self.model.features.children():
                ct += 1
                if ct < 5:
                    for param in child.parameters():
                        param.requires_grad = False

        elif model_name == 'efficientnet_b4':
            self.model = models.efficientnet_b4(pretrained=pretrained)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            # 冻结前8层
            ct = 0
            for child in self.model.features.children():
                ct += 1
                if ct < 5:
                    for param in child.parameters():
                        param.requires_grad = False

        # elif model_name == 'vgg16':
        #     self.model = models.vgg16(pretrained=pretrained)
        #     num_ftrs = self.model.classifier[6].in_features
        #     self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        #     # 冻结前12层
        #     for i, child in enumerate(self.model.features.children()):
        #         if i < 24:  # VGG16的前12个conv层
        #             for param in child.parameters():
        #                 param.requires_grad = False

        # 在BaseModel的__init__里新增：
        elif model_name == 'convnext_base':
            self.model = models.convnext_base(pretrained=True)
            num_ftrs = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(num_ftrs, Config.NUM_CLASSES)
            # 冻结前8层（和VGG冻结逻辑一致）
            ct = 0
            for child in self.model.features.children():
                ct += 1
                if ct < 8:
                    for param in child.parameters():
                        param.requires_grad = False

        elif model_name == 'inception_v3':
            self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            # 冻结前15层
            ct = 0
            for child in self.model.children():
                ct += 1
                if ct < 10:
                    for param in child.parameters():
                        param.requires_grad = False
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def forward(self, x):
        return self.model(x)


# ==================== 4. 训练函数 ====================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, model_name, patience=5):
    """训练单个基础模型"""
    print(f"\nTraining {model_name}...")

    best_val_f1 = 0.0
    best_model_wts = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        # 使用tqdm创建进度条，添加leave=False确保进度条完成后不保留
        loop = tqdm(train_loader, total=len(train_loader), leave=False,
                    desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs, labels in loop:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 更新进度条显示当前批次损失
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证阶段
        val_loss, val_acc, val_f1, _ = evaluate_model(model, val_loader, criterion)

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"Val F1: {val_f1:.4f}")

        # 早停检查
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 加载最佳模型权重
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, history, best_val_f1


def evaluate_model(model, data_loader, criterion):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 为评估过程也添加进度条
        loop = tqdm(data_loader, total=len(data_loader), leave=False, desc="Evaluating")
        for inputs, labels in loop:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条显示当前批次损失
            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, accuracy, f1_macro, (all_preds, all_labels)


def get_probabilities(model, data_loader):
    """获取模型输出的概率分布"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        # 为概率计算过程添加进度条
        loop = tqdm(data_loader, total=len(data_loader), leave=False, desc="Calculating probabilities")
        for inputs, labels in loop:
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.vstack(all_probs)
    return all_probs, np.array(all_labels)


# ==================== 5. 集成策略 ====================

class EnsembleModel:
    """集成模型类"""

    def __init__(self, base_models, ensemble_method='weighted_voting'):
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.weights = None
        self.meta_model = None

    def fit_weights(self, val_loader, val_f1_scores):
        """基于验证集F1分数计算权重"""
        total_f1 = sum(val_f1_scores)
        self.weights = [f1 / total_f1 for f1 in val_f1_scores]
        print(f"\nEnsemble weights: {self.weights}")

    def fit_stacking(self, train_loader, val_loader, train_labels, val_labels):
        """训练Stacking元模型"""
        print("\nTraining Stacking meta-model...")

        # 生成元特征
        train_meta_features = []
        val_meta_features = []

        for model_name, model in self.base_models:
            train_probs, _ = get_probabilities(model, train_loader)
            val_probs, _ = get_probabilities(model, val_loader)
            train_meta_features.append(train_probs)
            val_meta_features.append(val_probs)

        X_train_meta = np.hstack(train_meta_features)
        X_val_meta = np.hstack(val_meta_features)

        # 训练逻辑回归元模型
        self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_model.fit(X_train_meta, train_labels)

        # 验证元模型
        val_pred = self.meta_model.predict(X_val_meta)
        val_acc = accuracy_score(val_labels, val_pred)
        val_f1 = f1_score(val_labels, val_pred, average='macro')

        print(f"Stacking meta-model - Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    def predict_weighted_voting(self, data_loader):
        """加权软投票预测"""
        all_probs = []
        all_labels = []

        # 获取所有基础模型的概率
        for (model_name, model), weight in zip(self.base_models, self.weights):
            probs, labels = get_probabilities(model, data_loader)
            all_probs.append(probs * weight)
            if len(all_labels) == 0:
                all_labels = labels

        # 加权求和
        ensemble_probs = np.sum(all_probs, axis=0)
        predictions = np.argmax(ensemble_probs, axis=1)

        return predictions, all_labels

    def predict_stacking(self, data_loader):
        """Stacking预测"""
        # 生成元特征
        meta_features = []
        all_labels = None

        for model_name, model in self.base_models:
            probs, labels = get_probabilities(model, data_loader)
            meta_features.append(probs)
            if all_labels is None:
                all_labels = labels

        X_meta = np.hstack(meta_features)
        predictions = self.meta_model.predict(X_meta)

        return predictions, all_labels


# ==================== 6. 主流程 ====================

def main():
    print("=" * 60)
    print("Sea Animal Classification - Ensemble Model")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")

    # 1. 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx, class_weights = load_dataset()

    # 保存类别映射
    with open(os.path.join(Config.OUTPUT_DIR, 'class_to_idx.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)

    # 2. 定义基础模型配置
    base_model_configs = [
        ('resnet50', 5e-5, 'A'),
        ('mobilenet_v3', 1e-4, 'A'),
        ('efficientnet_b4', 3e-5, 'A'),
        ('convnext_base', 4e-5, 'B'),
        ('resnet50', 4e-5, 'B'),  # ResNet50副本,不同增强
    ]

    # 3. 训练基础模型
    trained_models = []
    val_f1_scores = []

    for model_name, lr, aug_group in base_model_configs:
        print(f"\n{'=' * 60}")
        print(f"Training {model_name} with augmentation group {aug_group}")
        print(f"{'=' * 60}")

        # 数据加载
        train_transform, val_transform = get_transforms(aug_group)

        train_dataset = SeaAnimalDataset(X_train, y_train, train_transform)
        val_dataset = SeaAnimalDataset(X_val, y_val, val_transform)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE,
                                shuffle=False, num_workers=4, pin_memory=True)

        # 创建模型
        model = BaseModel(model_name, Config.NUM_CLASSES).to(Config.DEVICE)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(Config.DEVICE))
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # 训练
        model, history, best_f1 = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            Config.NUM_EPOCHS, f"{model_name}_{aug_group}", patience=5
        )

        # 保存模型
        model_path = os.path.join(Config.MODEL_DIR, f"{model_name}_{aug_group}.pth")
        torch.save(model.state_dict(), model_path)

        trained_models.append((f"{model_name}_{aug_group}", model))
        val_f1_scores.append(best_f1)

    # 4. 集成模型 - 加权软投票
    print(f"\n{'=' * 60}")
    print("Ensemble Model - Weighted Voting")
    print(f"{'=' * 60}")

    _, val_transform = get_transforms('A')
    test_dataset = SeaAnimalDataset(X_test, y_test, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                             shuffle=False, num_workers=4)

    ensemble = EnsembleModel(trained_models, 'weighted_voting')
    ensemble.fit_weights(val_loader, val_f1_scores)

    # 测试集评估
    print("Evaluating ensemble model on test set...")
    predictions, true_labels = ensemble.predict_weighted_voting(test_loader)

    test_acc = accuracy_score(true_labels, predictions)
    test_f1_macro = f1_score(true_labels, predictions, average='macro')
    test_f1_micro = f1_score(true_labels, predictions, average='micro')

    print(f"\nWeighted Voting Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 (Macro): {test_f1_macro:.4f}")
    print(f"Test F1 (Micro): {test_f1_micro:.4f}")

    # 详细分类报告
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=target_names))

    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Ensemble Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    print(f"\nConfusion matrix saved to {Config.OUTPUT_DIR}/confusion_matrix.png")

    # 保存结果
    results = {
        'weighted_voting': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1_macro),
            'f1_micro': float(test_f1_micro)
        },
        'base_models_f1': {name: float(f1) for (name, _), f1 in zip(trained_models, val_f1_scores)}
    }

    with open(os.path.join(Config.OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"Models saved to: {Config.MODEL_DIR}")
    print(f"Results saved to: {Config.OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()