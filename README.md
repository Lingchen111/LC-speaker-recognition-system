# LC-speaker recognition system

一个基于深度学习的说话人识别系统，支持说话人标注和识别功能。

## 项目结构

```
LC-speaker recognition system/
├── configs/              # 配置文件目录
│   └── config.yaml       # 主配置文件
├── data/                 # 数据目录
│   ├── raw/              # 原始音频文件
│   ├── features/         # 提取的特征
│   └── annotations/      # 说话人标注
├── models/               # 模型目录
│   └── pretrained/       # 预训练模型
├── src/                  # 源代码目录
│   ├── annotation/       # 说话人标注模块
│   │   ├── __init__.py
│   │   └── annotator.py  # 标注器实现
│   ├── features/         # 特征提取模块
│   │   ├── __init__.py
│   │   └── audio_features.py  # 音频特征提取
│   ├── recognition/      # 说话人识别模块
│   │   ├── __init__.py
│   │   └── speaker_recognizer.py  # 说话人识别器实现
│   ├── __init__.py
│   └── main.py           # 主函数
├── run.py                # 运行脚本
├── requirements.txt      # 依赖项
├── LICENSE               # 许可证
└── .gitignore            # Git忽略文件
```

## 功能特性

- **说话人标注**：支持为音频文件添加说话人信息和标注
- **说话人识别**：支持基于特征向量的说话人识别和验证
- **特征提取**：使用先进的音频特征提取算法
- **模型支持**：集成了ECAPA-TDNN等先进模型
- **配置灵活**：通过配置文件调整系统参数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

### 1. 说话人标注模式

```bash
python run.py --mode annotate --audio <音频文件路径>
```

**示例**：
```bash
python run.py --mode annotate --audio data/raw/sample.wav
```

运行后会提示输入：
- 说话人姓名
- 性别 (M/F)
- 年龄
- 音频片段的起始时间（秒）
- 音频片段的结束时间（秒）
- 说话内容

### 2. 说话人识别模式

#### 基本识别（仅提取特征）
```bash
python run.py --mode recognize --audio <测试音频路径>
```

#### 说话人验证（与参考音频比较）
```bash
python run.py --mode recognize --audio <测试音频路径> --reference <参考音频路径>
```

**示例**：
```bash
python run.py --mode recognize --audio data/raw/test.wav --reference data/raw/reference.wav
```

## 配置说明

主要配置文件位于 `configs/config.yaml`，包含以下配置项：

- **data**: 数据目录和采样率配置
- **feature_extraction**: 特征提取参数（窗口大小、 hop大小、梅尔 bins等）
- **model**: 模型配置（名称、预训练路径、嵌入维度）
- **annotation**: 标注配置（片段时长、重叠等）
- **recognition**: 识别配置（阈值、最大说话人数、聚类方法等）

## 技术实现

- **特征提取**：使用Librosa库提取梅尔频谱特征
- **模型**：集成了ECAPA-TDNN模型用于说话人嵌入
- **相似度计算**：使用余弦相似度进行说话人验证
- **聚类**：支持K-means聚类方法

## 示例输出

### 说话人验证结果

```
=== 处理测试音频 ===
已成功提取特征向量，维度: (192,)

=== 进行说话人验证 ===

验证结果:
相似度: 0.8567
阈值: 0.6
是否为同一说话人: 是
```

## 注意事项

1. 音频文件格式建议使用WAV格式，采样率16kHz
2. 首次运行时会自动下载预训练模型
3. 标注功能需要手动输入说话人信息和片段信息
4. 识别功能的准确率取决于音频质量和模型训练数据

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请联系项目维护者。