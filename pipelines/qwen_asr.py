import torch
from qwen_asr import Qwen3ASRModel

class QwenASRPipeline():
    def __init__(self, device='npu:0'):
        self.model = Qwen3ASRModel.from_pretrained(
            "/models/Qwen/Qwen3-ASR-1.7B",
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            max_inference_batch_size=-1,
            max_new_tokens=256,
        )

    def chat_template(self, text, audio):
        return audio

    def run_batch(self, conversations):
        results = self.model.transcribe(
            audio=conversations,
            language="English",
        )

        return [res.text for res in results]