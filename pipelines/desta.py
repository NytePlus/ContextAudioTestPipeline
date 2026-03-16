try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

import torch
from pipelines.DeSTA25_Audio.desta import DeSTA25AudioModel

class DestaPipeline():
    def __init__(self, output='final'):
        self.model = DeSTA25AudioModel.from_pretrained("/models/NytePlus/DeSTA2.5-Audio-Llama-3.1-8B")
        self.model.to("cuda")
        self.output = output

    def chat_template(self, text, audio):
        conversation = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instruction. Respond directly without any other words"
            },
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{text}",
                "audios": [{
                    "audio": audio,
                    "text": None
                }]
            }
        ]
        return conversation

    def run_batch(self, conversations):
        outputs = self.model.generate(
            messages=conversations,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=512
        )

        if self.output == 'final':
            return [text for text in outputs.text]
        else:
            assert self.output == 'whisper'
            return [t for a, t in outputs.audios]