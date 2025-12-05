# ====================== 1. 导入必要库 ======================
import os
import json
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms


# ====================== 终端颜色美化工具 ======================
class Color:
    """终端颜色ANSI码"""
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"  # 重置颜色
    BOLD = "\033[1m"  # 加粗
    UNDERLINE = "\033[4m"  # 下划线


# ====================== 2. 复用核心组件 ======================
class Config:
    DATA_DIR = r"D:\pythondata\机器学习实验\fish_classification\Data"
    MODEL_DIR = r"D:\pythondata\机器学习实验\fish_classification\models"
    OUTPUT_DIR = r"D:\pythondata\机器学习实验\fish_classification\output"
    IMG_SIZE = 300
    BATCH_SIZE = 32
    NUM_CLASSES = 23
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_and_pad(image, target_size=Config.IMG_SIZE, pad_color=(0, 0, 0)):
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_image = Image.new(image.mode, (target_size, target_size), pad_color)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    new_image.paste(image, (offset_x, offset_y))
    return new_image


class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        from torchvision import models
        from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
        from torchvision.models import EfficientNet_B4_Weights, ConvNeXt_Base_Weights

        if model_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet50(weights=weights)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
            ct = 0
            for child in self.model.children():
                ct += 1
                if ct < 7:
                    for param in child.parameters():
                        param.requires_grad = False
        elif model_name == 'mobilenet_v3':
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.mobilenet_v3_large(weights=weights)
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)
            ct = 0
            for child in self.model.features.children():
                ct += 1
                if ct < 5:
                    for param in child.parameters():
                        param.requires_grad = False
        elif model_name == 'efficientnet_b4':
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_b4(weights=weights)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            ct = 0
            for child in self.model.features.children():
                ct += 1
                if ct < 6:
                    for param in child.parameters():
                        param.requires_grad = False
        elif model_name == 'convnext_base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_base(weights=weights)
            num_ftrs = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(num_ftrs, num_classes)
            ct = 0
            for child in self.model.features.children():
                ct += 1
                if ct < 8:
                    for param in child.parameters():
                        param.requires_grad = False
        else:
            raise ValueError(f"未知模型：{model_name}")

    def forward(self, x):
        return self.model(x)


class EnsembleModel:
    def __init__(self, trained_models, strategy='weighted_voting'):
        self.trained_models = trained_models
        self.strategy = strategy
        self.weights = None

    def set_weights(self, weights):
        assert len(weights) == len(self.trained_models), "权重数量和模型数量不匹配"
        self.weights = weights

    def predict_single_image(self, image_tensor):
        all_probs = []
        single_preds = []
        image_tensor = image_tensor.to(Config.DEVICE)

        for model_name, model in self.trained_models:
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)
                prob = torch.softmax(output, dim=1)
                all_probs.append(prob.cpu().numpy())
                pred_idx = torch.argmax(prob, dim=1).item()
                single_preds.append((model_name, pred_idx))

        all_probs = np.array(all_probs)
        if self.weights is None:
            self.weights = [1 / len(self.trained_models)] * len(self.trained_models)
        weighted_probs = np.sum(all_probs * np.array(self.weights).reshape(-1, 1, 1), axis=0)
        final_pred_idx = np.argmax(weighted_probs)
        final_pred_prob = weighted_probs[0][final_pred_idx]

        return final_pred_idx, final_pred_prob, single_preds


# ====================== 3. 加载类别映射 ======================
def load_class_mapping():
    class_to_idx_path = os.path.join(Config.OUTPUT_DIR, 'class_to_idx.json')
    if not os.path.exists(class_to_idx_path):
        class_names = [
            'Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 'Jelly Fish',
            'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 'Penguin', 'Puffers',
            'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Shrimp',
            'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale'
        ]
        class_to_idx = {name: i for i, name in enumerate(class_names)}
    else:
        with open(class_to_idx_path, 'r') as f:
            class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class


# ====================== 4. 加载集成模型 ======================
def load_ensemble_model():
    model_configs = [
        ('resnet50_A', 'resnet50'),
        ('mobilenet_v3_A', 'mobilenet_v3'),
        ('efficientnet_b4_A', 'efficientnet_b4'),
        ('convnext_base_B', 'convnext_base'),
        ('resnet50_B', 'resnet50')
    ]

    trained_models = []
    for weight_name, model_type in model_configs:
        model = BaseModel(model_type, Config.NUM_CLASSES, pretrained=False).to(Config.DEVICE)
        weight_path = os.path.join(Config.MODEL_DIR, f"{weight_name}.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在：{weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
        model.eval()
        trained_models.append((weight_name, model))

    ensemble = EnsembleModel(trained_models, strategy='weighted_voting')
    ensemble.set_weights([0.2, 0.2, 0.2, 0.2, 0.2])
    return ensemble


# ====================== 5. 单样本预测核心函数 ======================
def predict_single_image_ensemble(img_path, ensemble_model, idx_to_class):
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"加载图片失败：{e}")

    image = resize_and_pad(image, target_size=Config.IMG_SIZE)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = val_transform(image).unsqueeze(0)

    final_pred_idx, final_pred_prob, single_preds = ensemble_model.predict_single_image(image_tensor)
    final_pred_class = idx_to_class[final_pred_idx]

    single_pred_results = []
    for model_name, pred_idx in single_preds:
        pred_class = idx_to_class[pred_idx]
        single_pred_results.append({
            "model_name": model_name,
            "pred_class": pred_class,
            "pred_idx": pred_idx
        })

    return final_pred_class, final_pred_prob, single_pred_results


# ====================== 6. 运行单样本测试 ======================
if __name__ == "__main__":
    # 加载类别映射
    idx_to_class = load_class_mapping()
    print(f"{Color.BLUE}{'=' * 50}{Color.RESET}")
    print(f"{Color.BLUE}{'类别映射加载完成':^50}{Color.RESET}")
    print(f"{Color.BLUE}{f'总类别数: {len(idx_to_class)}':^50}{Color.RESET}")
    print(f"{Color.BLUE}{'=' * 50}{Color.RESET}\n")

    # 加载集成模型
    print(f"{Color.CYAN}🔧 正在加载集成模型...{Color.RESET}")
    ensemble_model = load_ensemble_model()
    print(f"{Color.GREEN}✅ 集成模型加载完成（包含 {len(ensemble_model.trained_models)} 个基础模型）{Color.RESET}\n")

    # 指定测试图片路径
    test_img_path = r"D:\pythondata\机器学习实验\fish_classification\鱿鱼squid.jpg"
    if not os.path.exists(test_img_path):
        raise FileNotFoundError(f"{Color.RED}❌ 测试图片不存在：{test_img_path}{Color.RESET}")

    # 预测
    img_name = os.path.basename(test_img_path)
    print(f"{Color.CYAN}🔍 正在预测图片: {img_name}{Color.RESET}")
    final_class, final_prob, single_results = predict_single_image_ensemble(
        test_img_path, ensemble_model, idx_to_class
    )

    # 终极美化输出
    print(f"\n{Color.PURPLE}{'=' * 50}{Color.RESET}")
    print(f"{Color.PURPLE}{Color.BOLD}{'集成模型最终预测结果':^50}{Color.RESET}")
    print(f"{Color.PURPLE}{'=' * 50}{Color.RESET}")
    # 最终结果（彩色高亮）
    print(f"📌 预测类别: {Color.GREEN}{Color.BOLD}{final_class:>20}{Color.RESET}")
    print(f"📊 置信度: {Color.YELLOW}{Color.BOLD}{f'{final_prob * 100:.2f}%':>22}{Color.RESET}")
    print(f"{Color.PURPLE}{'=' * 50}{Color.RESET}\n")

    # 基础模型结果（精准列宽表格）
    print(f"{Color.BLUE}{'基础模型预测详情':^50}{Color.RESET}")
    print(f"{Color.BLUE}{'-' * 50}{Color.RESET}")
    # 表头（固定列宽）
    header = f"{'模型名称':<18} | {'预测类别':<18} | {'类别索引'}"
    print(f"{Color.BOLD}{header}{Color.RESET}")
    print(f"{Color.BLUE}{'-' * 50}{Color.RESET}")
    # 内容行（严格对齐）
    for res in single_results:
        row = f"{res['model_name']:<18} | {res['pred_class']:<18} | {res['pred_idx']:>9}"
        print(row)
    print(f"{Color.BLUE}{'-' * 50}{Color.RESET}")