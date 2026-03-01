import torch
import yaml
from pathlib import Path
import logging
import numpy as np
from sklearn.cluster import KMeans
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置环境变量，避免符号链接问题
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_ENABLE_SYMLINKS'] = '0'

try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:
    logger.warning("SpeechBrain library not found. Please install it with: pip install speechbrain")


class SpeakerRecognizer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 设置模型保存目录
        self.model_dir = Path("models/pretrained/ecapa_tdnn")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载 SpeechBrain 的 SpeakerRecognition 模型
        try:
            self.verifier = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.model_dir),
                run_opts={"device": str(self.device)}
            )
            logger.info("Successfully loaded SpeakerRecognition model")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
        self.threshold = self.config['recognition']['threshold']
    
    def extract_embedding(self, audio_path):
        """提取说话人嵌入向量"""
        try:
            # 使用 SpeakerRecognition 模型的 encode_batch 方法
            wav = self.verifier.load_audio(audio_path)
            wav = wav.unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.verifier.encode_batch(wav)
            
            return embedding[0][0].cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            logger.error(f"Error details: {str(e.__class__.__name__)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def compute_similarity(self, embedding1, embedding2):
        """计算两个嵌入向量的余弦相似度"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def verify_speaker(self, audio_path1, audio_path2):
        """使用官方方法直接验证说话人"""
        try:
            score, prediction = self.verifier.verify_files(audio_path1, audio_path2)
            
            return {
                'is_same_speaker': bool(prediction),
                'similarity': float(score),
                'threshold': float(self.verifier.threshold) if hasattr(self.verifier, 'threshold') else self.threshold
            }
        except Exception as e:
            logger.error(f"Error verifying speaker: {str(e)}")
            # 如果官方方法失败，回退到我们自己的方法
            logger.warning("Falling back to custom verification method")
            embedding1 = self.extract_embedding(audio_path1)
            embedding2 = self.extract_embedding(audio_path2)
            if embedding1 is not None and embedding2 is not None:
                similarity = self.compute_similarity(embedding1, embedding2)
                is_same_speaker = similarity >= self.threshold
                return {
                    'is_same_speaker': is_same_speaker,
                    'similarity': float(similarity),
                    'threshold': self.threshold
                }
            return None
    
    def cluster_speakers(self, embeddings, n_clusters=None):
        """说话人聚类"""
        if not self.config['recognition']['use_clustering']:
            return None
        
        if n_clusters is None:
            n_clusters = min(len(embeddings), self.config['recognition']['max_speakers'])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        return {
            'labels': labels,
            'centroids': kmeans.cluster_centers_
        }
    
    def process_audio(self, audio_path, reference_embeddings=None):
        """处理音频文件"""
        embedding = self.extract_embedding(audio_path)
        if embedding is None:
            return None
        
        result = {
            'embedding': embedding,
            'audio_path': str(audio_path)
        }
        
        # 如果提供了参考嵌入向量，进行说话人验证
        if reference_embeddings is not None:
            verifications = []
            for ref_name, ref_embedding in reference_embeddings.items():
                similarity = self.compute_similarity(ref_embedding, embedding)
                is_same_speaker = similarity >= self.threshold
                verifications.append({
                    'reference_name': ref_name,
                    'is_same_speaker': is_same_speaker,
                    'similarity': float(similarity),
                    'threshold': self.threshold
                })
            result['verifications'] = verifications
        
        return result

def main():
    """测试说话人识别"""
    config_path = "configs/config.yaml"
    recognizer = SpeakerRecognizer(config_path)
    
    # 测试音频处理
    test_audio = "data/raw/test.wav"
    if Path(test_audio).exists():
        result = recognizer.process_audio(test_audio)
        if result:
            logger.info(f"Successfully processed {test_audio}")
            logger.info(f"Embedding shape: {result['embedding'].shape}")
        else:
            logger.error(f"Failed to process {test_audio}")
    else:
        logger.warning(f"Test audio file {test_audio} does not exist")

if __name__ == "__main__":
    main() 