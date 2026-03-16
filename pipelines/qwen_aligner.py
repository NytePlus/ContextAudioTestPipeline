import torch
from qwen_asr import Qwen3ForcedAligner

class QwenAlignerPipeline():
    def __init__(self):
        self.model = Qwen3ForcedAligner.from_pretrained(
            "/models/Qwen/Qwen3-ForcedAligner-0.6B",
            dtype=torch.bfloat16,
            device_map="npu:6",
            attn_implementation="flash_attention_2",
        )

    def chat_template(self, text, audio):
        return {
            'audio': audio,
            'text': text
        }

    def run_batch(self, conversations):
        audios = [c['audio'] for c in conversations]
        text = [c['text'] for c in conversations]
        results = self.model.align(
            audio=audios,
            text=text,
            language=["English"] * len(conversations),
        )

        return [res for res in results]