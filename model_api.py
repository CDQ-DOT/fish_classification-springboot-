import os
import json
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from test import EnsembleModel, BaseModel, Config, resize_and_pad, load_class_mapping  # 导入必要工具

# 初始化FastAPI应用
app = FastAPI(title="模型预测API")

# 跨域配置（允许SpringBoot的8080端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # SpringBoot端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量：加载集成模型和类别映射（启动时只加载一次）
ensemble_model = None
idx_to_class = None  # 类别索引到名称的映射


def get_base_models_f1():
    """从results.json读取各基础模型的F1分数"""
    results_path = os.path.join(Config.OUTPUT_DIR, "results.json")  # 修正路径为OUTPUT_DIR
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"结果文件不存在：{results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    # 按model_configs的顺序提取F1（需和load_ensemble_model中的模型顺序一致）
    f1_scores = [
        results.get('base_models_f1', {}).get('resnet50_A', 0.0),
        results.get('base_models_f1', {}).get('mobilenet_v3_A', 0.0),
        results.get('base_models_f1', {}).get('efficientnet_b4_A', 0.0),
        results.get('base_models_f1', {}).get('convnext_base_B', 0.0),
        results.get('base_models_f1', {}).get('resnet50_B', 0.0)
    ]
    if any(f == 0.0 for f in f1_scores):
        raise ValueError("部分模型的F1分数未找到，请检查results.json格式")
    return f1_scores


def load_ensemble_model():
    """加载集成模型（启动时执行）"""
    global ensemble_model, idx_to_class
    if ensemble_model is not None:
        return ensemble_model

    # 加载类别映射
    idx_to_class = load_class_mapping()

    # 1. 模型配置（和训练的模型顺序一致）
    model_configs = [
        ('resnet50_A', 'resnet50'),
        ('mobilenet_v3_A', 'mobilenet_v3'),
        ('efficientnet_b4_A', 'efficientnet_b4'),
        ('convnext_base_B', 'convnext_base'),
        ('resnet50_B', 'resnet50')
    ]

    # 2. 加载基础模型
    trained_models = []
    for weight_name, model_type in model_configs:
        model = BaseModel(model_name=model_type, num_classes=Config.NUM_CLASSES, pretrained=False).to(Config.DEVICE)
        weight_path = os.path.join(Config.MODEL_DIR, f"{weight_name}.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"权重文件不存在：{weight_path}")
        # 加载权重（兼容CPU/GPU）
        model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
        model.eval()  # 评估模式
        trained_models.append((weight_name, model))

    # 3. 读取F1并计算权重
    val_f1_scores = get_base_models_f1()
    weights = np.array(val_f1_scores) / sum(val_f1_scores)  # 归一化

    # 4. 初始化集成模型
    ensemble_model = EnsembleModel(
        trained_models=trained_models,  # 修正参数名与test.py一致
        strategy='weighted_voting'
    )
    ensemble_model.set_weights(weights.tolist())  # 设置权重
    return ensemble_model


# 启动时加载模型
@app.on_event("startup")
async def startup_event():
    try:
        load_ensemble_model()
        print("集成模型加载成功！")
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{str(e)}")


# 预测接口（接收图片文件）
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. 读取并处理图片
        image = Image.open(file.file).convert('RGB')
        image = resize_and_pad(image, target_size=Config.IMG_SIZE)  # 等比例缩放+填充

        # 2. 图片转换为Tensor
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = val_transform(image).unsqueeze(0)  # 添加batch维度

        # 3. 调用集成模型预测
        final_pred_idx, final_pred_prob, single_preds = ensemble_model.predict_single_image(image_tensor)

        # 4. 转换结果为类别名称
        final_pred_class = idx_to_class[final_pred_idx]
        single_results = []
        for model_name, pred_idx in single_preds:
            single_results.append({
                "model_name": model_name,
                "pred_class": idx_to_class[pred_idx],
                "pred_idx": pred_idx
            })

        return {
            "code": 200,
            "msg": "预测成功",
            "data": {
                "final_pred_class": final_pred_class,
                "final_pred_prob": float(final_pred_prob),  # 转换为Python float
                "single_model_results": single_results
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)