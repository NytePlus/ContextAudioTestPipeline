import torch
import warnings
from bisect import bisect_left
warnings.filterwarnings("ignore")

from transformers import AutoProcessor
from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

class MingOmni():
    def __init__(self, model_path="/models/inclusionAI/Ming-flash-omni-Preview"):
        def split_model():
            device_map = {}
            world_size = torch.npu.device_count()
            num_layers = 32
            layer_per_gpu = num_layers // world_size
            layer_per_gpu = [i * layer_per_gpu for i in range(1, world_size + 1)]
            for i in range(num_layers):
                device_map[f'model.model.layers.{i}'] = bisect_left(layer_per_gpu, i)
            device_map['vision'] = 0
            device_map['audio'] = 0
            device_map['linear_proj'] = 0
            device_map['linear_proj_audio'] = 0
            device_map['model.model.word_embeddings.weight'] = 0
            device_map['model.model.norm.weight'] = 0
            device_map['model.lm_head.weight'] = 0
            device_map['model.model.norm'] = 0
            device_map[f'model.model.layers.{num_layers - 1}'] = 0
            return device_map

        # Load pre-trained model with optimized settings, this will take ~10 minutes
        self.model = BailingMM2NativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=split_model(),
            load_image_gen=True,
            load_talker=True,
        ).to(dtype=torch.bfloat16)

        # Initialize processor for handling multimodal inputs
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def chat_template(self, text, image, audio):
        conversation = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "audio": audio},
                    {"type": "image", "image": image}
                ],
            },
        ]
        return conversation

    # Inference Pipeline
    def run_batch(self, conversations, sys_prompt_exp=None, use_cot_system_prompt=False, max_new_tokens=512):
        text = self.processor.apply_chat_template(
            conversations, 
            sys_prompt_exp=sys_prompt_exp,
            use_cot_system_prompt=use_cot_system_prompt
        )
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(conversations)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            audio_kwargs={"use_whisper_encoder": True},
        ).to(self.model.device)

        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=self.processor.gen_terminator,
                num_logits_to_keep=1,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts
