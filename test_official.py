import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_official_verifier():
    """使用官方 SpeakerRecognition 测试"""
    try:
        from src.recognition.speaker_recognizer import SpeakerRecognizer
        
        logger.info("初始化 SpeakerRecognizer...")
        config_path = "configs/config.yaml"
        
        recognizer = SpeakerRecognizer(config_path)
        logger.info("✅ 模型加载成功！")
        
        # 定义音频文件对
        test_pairs = [
            ("test3.wav", "test4.wav")
        ]
        
        # 检查文件是否存在
        for a1, a2 in test_pairs:
            path1 = f"data/raw/{a1}"
            path2 = f"data/raw/{a2}"
            if not Path(path1).exists():
                logger.error(f"❌ 文件不存在: {path1}")
                return False
            if not Path(path2).exists():
                logger.error(f"❌ 文件不存在: {path2}")
                return False
        
        logger.info("\n" + "="*70)
        logger.info("使用官方 SpeakerRecognition 方法测试")
        logger.info("="*70)
        
        for a1, a2 in test_pairs:
            path1 = f"data/raw/{a1}"
            path2 = f"data/raw/{a2}"
            
            logger.info(f"\n测试: {a1} vs {a2}")
            
            result = recognizer.verify_speaker(path1, path2)
            
            if result:
                logger.info(f"  相似度: {result['similarity']:.4f}")
                logger.info(f"  阈值: {result['threshold']}")
                logger.info(f"  是否为同一说话人: {'✅ 是' if result['is_same_speaker'] else '❌ 否'}")
            else:
                logger.error("  ❌ 验证失败")
        
        logger.info("\n" + "="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_official_verifier()
    sys.exit(0 if success else 1)