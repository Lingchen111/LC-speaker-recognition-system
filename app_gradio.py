import gradio as gr
import os
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量，用于存储识别器
recognizer = None
model_loaded = False

def load_model():
    """加载模型"""
    global recognizer, model_loaded
    
    if model_loaded:
        return "✅ 模型已加载", gr.update(interactive=False), gr.update(interactive=True)
    
    try:
        logger.info("正在初始化模型...")
        from src.recognition.speaker_recognizer import SpeakerRecognizer
        
        logger.info("正在加载模型...")
        recognizer = SpeakerRecognizer("configs/config.yaml")
        model_loaded = True
        
        logger.info("✅ 模型加载成功！")
        return "✅ 模型加载成功！", gr.update(interactive=False), gr.update(interactive=True)
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"❌ 模型加载失败: {str(e)}", gr.update(interactive=True), gr.update(interactive=False)

def verify_speakers(audio1, audio2):
    """验证两个音频是否为同一说话人"""
    global recognizer, model_loaded
    
    if not model_loaded or recognizer is None:
        return None, None, None, "请先点击 '加载模型' 按钮"
    
    try:
        # 处理音频文件
        if audio1 is None or audio2 is None:
            return None, None, None, "请上传两个音频文件或录制语音"
        
        # 保存临时文件到 data/raw 目录
        def save_temp_audio(audio_data):
            if audio_data is None:
                return None
            
            # 确保 data/raw 目录存在
            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成唯一的文件名
            import uuid
            filename = f"temp_{uuid.uuid4().hex}.wav"
            temp_path = str(raw_dir / filename)
            
            try:
                # 处理不同类型的音频数据
                if isinstance(audio_data, tuple):
                    # Gradio 麦克风录制的音频 (采样率, 数据)
                    sr, data = audio_data
                    sf.write(temp_path, data, sr)
                    logger.info(f"保存麦克风录制的音频到: {temp_path}")
                else:
                    # 上传的文件路径
                    if os.path.exists(audio_data):
                        # 直接复制文件到目标路径
                        import shutil
                        shutil.copy2(audio_data, temp_path)
                        logger.info(f"复制上传的文件到: {temp_path}")
                    else:
                        logger.warning(f"文件不存在: {audio_data}")
                        return None
            except Exception as e:
                logger.error(f"处理音频数据失败: {str(e)}")
                return None
            
            return temp_path
        
        # 保存两个音频文件
        path1 = save_temp_audio(audio1)
        path2 = save_temp_audio(audio2)
        
        if not path1 or not path2:
            return None, None, None, "音频文件处理失败"
        
        # 调用验证方法
        logger.info(f"验证音频: {path1} 和 {path2}")
        
        # 检查文件是否存在
        if not Path(path1).exists():
            logger.error(f"❌ 文件不存在: {path1}")
            return None, None, None, f"文件不存在: {path1}"
        if not Path(path2).exists():
            logger.error(f"❌ 文件不存在: {path2}")
            return None, None, None, f"文件不存在: {path2}"
        
        # 直接调用验证方法
        result = recognizer.verify_speaker(path1, path2)
        
        # 清理临时文件
        try:
            if path1 and os.path.exists(path1):
                os.unlink(path1)
            if path2 and os.path.exists(path2):
                os.unlink(path2)
        except Exception as e:
            logger.warning(f"清理临时文件失败: {str(e)}")
        
        if result:
            similarity = result['similarity']
            threshold = result['threshold']
            is_same = result['is_same_speaker']
            
            # 生成结果信息
            status = "✅ 是同一说话人" if is_same else "❌ 不是同一说话人"
            confidence = similarity * 100
            
            return status, f"{similarity:.4f}", f"{threshold}", ""
        else:
            return None, None, None, "验证失败，请检查音频文件"
            
    except Exception as e:
        logger.error(f"验证失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, f"验证过程中出错: {str(e)}"

def main():
    """创建 Gradio 界面"""
    # 创建界面
    with gr.Blocks(
        title="说话人识别系统", 
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate"
        ),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        .header-title {
            text-align: center !important;
            font-weight: bold !important;
            margin-bottom: 20px !important;
        }
        .result-card {
            padding: 20px !important;
            border-radius: 10px !important;
            margin-top: 20px !important;
        }
        .model-status {
            padding: 10px !important;
            border-radius: 8px !important;
            margin-bottom: 20px !important;
            text-align: center !important;
            font-weight: bold !important;
        }
        """
    ) as app:
        # 标题区域
        with gr.Row():
            gr.Markdown("""
            <div class="header-title">
            <h1 style="margin-bottom: 10px;">🎤 说话人识别系统</h1>
            <p style="color: #64748b; font-size: 16px;">基于 ECAPA-TDNN 模型的说话人验证工具</p>
            </div>
            """)
        
        # 模型加载区域
        with gr.Row():
            with gr.Column():
                model_status = gr.Textbox(
                    value="⏳ 等待加载模型...",
                    label="模型状态",
                    interactive=False,
                    show_label=True,
                    elem_classes="model-status"
                )
        
        with gr.Row():
            load_model_btn = gr.Button(
                "📥 加载模型", 
                variant="secondary", 
                size="lg"
            )
        
        # 使用说明 - 放在可折叠区域
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 功能介绍
            本系统使用预训练的 ECAPA-TDNN 模型进行说话人验证，支持两种输入方式：
            1. **上传 WAV 音频文件**
            2. **通过浏览器直接录制语音**
            
            ### 操作步骤
            1. 点击 "加载模型" 按钮（首次运行需要下载模型）
            2. 选择两个音频输入（上传文件或录制语音）
            3. 点击 "验证说话人" 按钮
            4. 查看验证结果
            
            ### 注意事项
            - 首次运行需要从 HuggingFace Hub 下载模型（约100MB）
            - 请确保网络连接正常
            - 音频文件最好使用 16kHz 采样率的 WAV 格式
            """)
        
        gr.Markdown("---")
        
        # 音频输入区域
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔊 第一个说话人")
                audio1 = gr.Audio(
                    sources=["upload", "microphone"], 
                    type="filepath",
                    label="音频输入",
                    show_download_button=True
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 🔊 第二个说话人")
                audio2 = gr.Audio(
                    sources=["upload", "microphone"], 
                    type="filepath",
                    label="音频输入",
                    show_download_button=True
                )
        
        with gr.Row():
            submit_btn = gr.Button(
                "✨ 验证说话人", 
                variant="primary", 
                size="lg",
                scale=1,
                interactive=False
            )
        
        gr.Markdown("---")
        
        # 结果显示区域
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 验证结果")
                with gr.Row():
                    with gr.Column(scale=2):
                        status_output = gr.Textbox(
                            label="验证状态",
                            placeholder="等待验证...",
                            lines=1,
                            interactive=False,
                            show_label=True
                        )
                    with gr.Column(scale=1):
                        similarity_output = gr.Textbox(
                            label="相似度",
                            placeholder="-",
                            lines=1,
                            interactive=False,
                            show_label=True
                        )
                    with gr.Column(scale=1):
                        threshold_output = gr.Textbox(
                            label="阈值",
                            placeholder="-",
                            lines=1,
                            interactive=False,
                            show_label=True
                        )
                error_output = gr.Textbox(
                    label="提示信息",
                    placeholder="",
                    lines=2,
                    interactive=False,
                    show_label=True
                )
        
        # 加载模型事件
        load_model_btn.click(
            fn=load_model,
            inputs=[],
            outputs=[model_status, load_model_btn, submit_btn]
        )
        
        # 点击事件
        submit_btn.click(
            fn=verify_speakers,
            inputs=[audio1, audio2],
            outputs=[status_output, similarity_output, threshold_output, error_output]
        )
    
    # 启动界面
    app.launch(
        server_name="127.0.0.1",
        server_port=7868,
        share=False
    )

if __name__ == "__main__":
    main()
