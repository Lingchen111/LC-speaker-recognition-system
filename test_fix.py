import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_speaker_recognizer():
    """测试说话人识别器"""
    try:
        from src.recognition.speaker_recognizer import SpeakerRecognizer
        
        logger.info("初始化 SpeakerRecognizer...")
        config_path = Path("configs/config.yaml")
        
        if config_path.exists():
            logger.info(f"配置文件 {config_path} 存在，开始加载模型...")
            recognizer = SpeakerRecognizer(str(config_path))
            logger.info("✅ 模型加载成功！")
            
            # 检查测试音频是否存在
            test_audio = "data/raw/test.wav"
            if Path(test_audio).exists():
                logger.info(f"测试音频文件存在，开始提取嵌入向量...")
                result = recognizer.process_audio(test_audio)
                if result:
                    logger.info(f"✅ 成功处理音频！嵌入向量形状: {result['embedding'].shape}")
                    return True
                else:
                    logger.error("❌ 音频处理失败")
                    return False
            else:
                logger.warning(f"测试音频文件 {test_audio} 不存在，但模型加载成功")
                return True
        else:
            logger.error(f"配置文件 {config_path} 不存在")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_speaker_recognizer()
    sys.exit(0 if success else 1)