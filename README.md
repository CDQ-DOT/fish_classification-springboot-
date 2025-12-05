# OceanAnimalClassifier - 海洋动物图片AI分类系统

一个集**海洋科技风前端界面**、**SpringBoot后端转发**、**FastAPI深度学习服务**于一体的海洋动物图片分类项目，支持23种海洋动物的自动识别，兼具视觉美感与功能实用性。


## 项目简介
本项目通过「前端可视化交互 + 后端接口转发 + 深度学习模型预测」的架构，实现海洋动物图片的快速分类：
- 前端：采用「深海渐变+动态波浪+毛玻璃卡片」设计，兼具海洋氛围与科技感；
- 后端：SpringBoot负责接口转发，FastAPI承载深度学习模型服务，实现功能解耦；
- 模型：基于PyTorch构建多基础模型（ResNet50/MobileNetV3等）的集成系统，提升分类准确性。


## 技术栈
| 模块         | 技术/工具                          | 核心作用                     |
|--------------|------------------------------------|------------------------------|
| 前端         | 原生HTML/CSS/JS                    | 海洋科技风交互界面           |
| 后端转发     | SpringBoot 2.7.x                   | 接收前端请求，转发至模型服务 |
| 模型服务     | FastAPI、Python 3.8+               | 承载深度学习模型，处理预测请求 |
| 深度学习     | PyTorch、TorchVision               | 多基础模型集成（加权投票策略） |


## 核心功能
1. **图片上传**：支持JPG/PNG格式的海洋动物图片上传；
2. **AI分类预测**：基于集成模型识别23种海洋动物；
3. **结果可视化**：展示预测类别、置信度进度条；
4. **响应式适配**：自动兼容PC/移动端设备。


## 快速开始
### 1. 环境准备
- Python 3.8+（用于模型服务）
- Java 8+ & Maven（用于SpringBoot后端）


### 2. 启动FastAPI模型服务
```bash
# 进入模型服务目录（包含model_api.py、test.py等）
cd 模型服务目录

# 安装依赖
pip install -r requirements.txt

# 启动模型服务（默认端口：8000）
python model_api.py

# 进入SpringBoot项目根目录
cd fish_classification

# 编译打包
mvn clean package

# 启动后端服务（默认端口：8080）
java -jar target/fish_classification-0.0.1-SNAPSHOT.jar

打开浏览器，访问：http://localhost:8080

这个是java的项目结构
fish_classification/
├── src/
│   ├── main/
│   │   ├── java/org/example/
│   │   │   ├── controller/       # SpringBoot接口转发Controller
│   │   │   ├── AppConfig.java    # 配置类（RestTemplate等）
│   │   │   └── FishClassificationApplication.java  # 启动类
│   │   └── resources/
│   │       └── static/
│   │           └── index.html    # 海洋科技风前端界面
├── model_api.py                  # FastAPI模型服务入口
├── test.py                       # 集成模型定义
├── requirements.txt              # Python依赖清单
└── README.md                     # 项目说明文档
