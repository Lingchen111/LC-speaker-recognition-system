# LC-speaker recognition system

一个基于 SpeechBrain 的说话人识别系统，使用预训练的 ECAPA-TDNN 模型进行说话人验证。

## 项目结构

```
LC-speaker recognition system/
├── configs/                  # 配置文件目录
│   └── config.yaml           # 主配置文件
├── data/                     # 数据目录
│   └── raw/                  # 原始音频文件
├── models/                   # 模型目录
│   └── pretrained/           # 预训练模型
│       └── ecapa_tdnn/      # ECAPA-TDNN 预训练模型
├── src/                      # 源代码目录
│   └── recognition/          # 说话人识别模块
│       ├── __init__.py
│       └── speaker_recognizer.py  # 说话人识别器实现
├── test_official.py          # 官方方法测试脚本
├── requirements.txt          # 依赖项
├── LICENSE                   # 许可证
└── .gitignore                # Git忽略文件
```

## 功能特性

- **说话人验证**：使用 SpeechBrain 官方预训练的 ECAPA-TDNN 模型验证两个音频是否为同一说话人
- **预训练模型**：自动从 Hugging Face Hub 下载 `speechbrain/spkrec-ecapa-voxceleb` 模型
- **配置灵活**：通过配置文件调整阈值等参数
- **简单易用**：提供简洁的 API 接口

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 使用测试脚本

项目提供了 `test_official.py` 测试脚本，可以快速测试说话人验证功能：

```bash
python test_official.py
```

### 在代码中使用

```python
from src.recognition.speaker_recognizer import SpeakerRecognizer

# 初始化识别器
recognizer = SpeakerRecognizer("configs/config.yaml")

# 验证两个音频是否为同一说话人
result = recognizer.verify_speaker("data/raw/audio1.wav", "data/raw/audio2.wav")

print(f"相似度: {result['similarity']:.4f}")
print(f"阈值: {result['threshold']}")
print(f"是否为同一说话人: {'是' if result['is_same_speaker'] else '否'}")
```

## 配置说明

主要配置文件位于 `configs/config.yaml`，包含以下配置项：

```yaml
data:
  raw_audio_dir: "data/raw"          # 原始音频目录
  sample_rate: 16000                  # 采样率

recognition:
  threshold: 0.6                       # 相似度阈值（0.6 为推荐值）
  max_speakers: 100                    # 最大说话人数
  use_clustering: true                  # 是否使用聚类
  clustering_method: "kmeans"          # 聚类方法
```

## 阈值调整

阈值的选择会影响识别结果：

- **阈值过高**：可能会将同一说话人误判为不同人（漏检）
- **阈值过低**：可能会将不同说话人误判为同一人（误检）

推荐阈值范围：0.5 - 0.7，可根据实际使用场景调整。

## 技术实现

- **模型**：使用 SpeechBrain 官方预训练的 ECAPA-TDNN 模型
- **特征提取**：模型内部使用 Fbank（Filter Bank）特征
- **相似度计算**：余弦相似度
- **推理**：首次运行时自动从 Hugging Face Hub 下载模型

## 示例输出

### 说话人验证结果

```
======================================================================
使用官方 SpeakerRecognition 方法测试
======================================================================

测试: test1.wav vs test2.wav
  相似度: 0.6369
  阈值: 0.6
  是否为同一说话人: ✅ 是

测试: test1.wav vs test3.wav
  相似度: 0.1146
  阈值: 0.6
  是否为同一说话人: ❌ 否

测试: test3.wav vs test4.wav
  相似度: 0.9471
  阈值: 0.6
  是否为同一说话人: ✅ 是
======================================================================
```

## 注意事项

1. 音频文件格式建议使用 WAV 格式，采样率 16kHz
2. 首次运行时会自动从 Hugging Face Hub 下载预训练模型（约 100MB）
3. 识别功能的准确率取决于音频质量
4. Windows 用户首次运行时可能会有符号链接警告，不影响使用

## 模型来源

本项目使用的预训练模型来自：
- **模型名称**：speechbrain/spkrec-ecapa-voxceleb
- **来源**：Hugging Face Hub
- **链接**：https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 联系方式

如有问题或建议，请联系项目维护者。
