import os
import shutil
from pathlib import Path

# 清理 HuggingFace 缓存
cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--speechbrain--spkrec-ecapa-voxceleb"
if cache_dir.exists():
    print(f"清理缓存: {cache_dir}")
    shutil.rmtree(cache_dir)
    print("缓存已清理")
else:
    print("缓存不存在，跳过清理")

# 现在测试模型加载
print("\n开始测试模型加载...")
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.recognition.speaker_recognizer import SpeakerRecognizer
    
    logger.info("初始化 SpeakerRecognizer...")
    config_path = Path("configs/config.yaml")
    
    if config_path.exists():
        logger.info(f"配置文件 {config_path} 存在，开始加载模型...")
        recognizer = SpeakerRecognizer(str(config_path))
        logger.info("✅ 模型加载成功！")
        print("\n🎉 测试成功！模型已正确加载！")
    else:
        logger.error(f"配置文件 {config_path} 不存在")
        
except Exception as e:
    logger.error(f"❌ 测试失败: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())