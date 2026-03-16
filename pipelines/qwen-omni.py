from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

USE_AUDIO_IN_VIDEO=False

class QwenOmniPipeline():
    def __init__(self,):
        model_name = "/models/Qwen/Qwen2.5-Omni-3B"
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="npu:0"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    def chat_template(self, text, image, audio):
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [],
            },
        ]

        conversation[1]["content"].append({"type": "text", "text": text})
        if image != '':
            conversation[1]["content"].append({"type": "image", "image": image})
        conversation[1]["content"].append({"type": "audio", "audio": audio})
        return conversation

    def run_batch(self, conversations, items):
        # 1. 构造 batch text
        texts = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False
        )

        # 2. 处理多模态输入
        audios, images, videos = process_mm_info(
            conversations,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )

        inputs = self.processor(
            text=texts,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # 3. 推理
        text_ids = self.model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            return_audio=False,
        )
        # 4. 解码
        outputs = self.processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return outputs