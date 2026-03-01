import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_methods():
    """检查模型的可用方法"""
    try:
        from src.recognition.speaker_recognizer import SpeakerRecognizer
        
        logger.info("初始化 SpeakerRecognizer...")
        config_path = "configs/config.yaml"
        
        recognizer = SpeakerRecognizer(config_path)
        logger.info("✅ 模型加载成功！")
        
        # 查看模型有哪些方法
        logger.info("\n模型可用的方法:")
        for method in dir(recognizer.model):
            if not method.startswith('_'):
                logger.info(f"  - {method}")
        
        # 查看 encode_batch 是否存在
        if hasattr(recognizer.model, 'encode_batch'):
            logger.info("\n✅ encode_batch 方法存在！")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = check_model_methods()
    sys.exit(0 if success else 1)