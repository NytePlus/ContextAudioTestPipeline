import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

class Qwen2AudioPipeline:
    def __init__(self, model_path="/home/ma-user/work/output/v3-20260315-154403/checkpoint-5388"):
        self.device = "npu:0"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, 
            device_map=self.device,
            torch_dtype="auto" # 建议使用自动精度以优化性能
        )

    def chat_template(self, text, audio_path):
        """
        构造符合 Qwen2-Audio 格式的对话结构
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": text},
                ],
            },
        ]
        return conversation

    def _load_audio(self, conversations):
        """
        内部方法：遍历对话并使用 librosa 加载音频数据
        """
        audios = []
        for message in conversations:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        # 使用 processor 预设的采样率 (通常是 16000)
                        audio_array, _ = librosa.load(
                            ele['audio'], 
                            sr=self.processor.feature_extractor.sampling_rate
                        )
                        audios.append(audio_array)
        return audios

    def run_batch(self, conversations):
        """
        执行推理流程
        注意：Qwen2-Audio 的输入通常是一个 conversation list (单个 batch) 
        或者 list of conversations (多个 batch)
        """
        # 1. 构造 prompt 文本
        # 处理单条或多条对话
        texts = self.processor.apply_chat_template(
            conversations, 
            add_generation_prompt=True, 
            tokenize=False
        )

        # 2. 处理多模态输入 (音频加载)
        # 如果是单条对话转为列表处理
        if isinstance(conversations[0], dict):
            input_conversations = [conversations]
        else:
            input_conversations = conversations
            
        all_audios = []
        for conv in input_conversations:
            all_audios.extend(self._load_audio(conv))

        # 3. 编码 inputs
        inputs = self.processor(
            text=texts, 
            audio=all_audios, 
            return_tensors="pt", 
            padding=True
        )
        
        # 将所有 tensor 移动到 NPU
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # 4. 开启推理
        generate_ids = self.model.generate(**inputs, max_length=2048)
        
        # 5. 只截取生成的回答部分（去掉 prompt 部分）
        # 针对 batch 处理，我们需要逐条截断
        input_len = inputs["input_ids"].size(1)
        generate_ids = generate_ids[:, input_len:]

        # 6. 解码
        outputs = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        return outputs