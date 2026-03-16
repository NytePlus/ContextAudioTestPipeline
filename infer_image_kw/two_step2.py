import json
import os
import re
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from data import Data
from template.prompt import cot_user_prompt_step1, cot_user_prompt_step2_no_exp, default_sys_prompt
from utils import hotword_process

data_dir = '/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech'
image_json = os.path.join(data_dir, "test_oracle_v1/slides/multitask.jsonl")  # 输入文件
kw_json = os.path.join(data_dir, "test_oracle_v1/hotword/multitask.jsonl")  # 输入文件
output_pred_path = "aikcot_test_pred"
USE_AUDIO_IN_VIDEO = False

model_name = "/models/Qwen/Qwen2.5-Omni-7B"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="npu:0",
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

data = Data(image_json, data_dir)
kw = Data(kw_json, data_dir)

def extract_asr_text(raw_text: str) -> str:
    """
    从 raw_text 中提取第一个包含 "asr_text" 的 JSON 对象，并返回其值
    """
    # 匹配所有 {} 包裹的 JSON 对象
    matches = re.findall(r'\{[\s\S]*?\}', raw_text)
    if not matches:
        raise ValueError("No JSON object found in raw_text")
    
    for json_str in matches:
        print(json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            continue  # 不是合法 JSON，跳过
        if "asr_text" in data:
            return data["asr_text"]
    
    raise ValueError('No JSON object with key "asr_text" found')

# ---------------------------
# 批量推理（两阶段在同一个上下文）
# ---------------------------
for image_item, kw_item in tqdm(zip(data, kw), desc="推理中"):
    kws = hotword_process(kw_item["hotword"])

    # 初始化 conversation
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": default_sys_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": cot_user_prompt_step1.format(kws)},
                {"type": "image", "image": image_item['ppt_path']},
            ],
        },
    ]

    # Step 1: 图片 + 关键字推理
    text_step1 = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios_step1, images_step1, videos_step1 = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs_step1 = processor(
        text=text_step1,
        audio=audios_step1,
        images=images_step1,
        videos=videos_step1,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs_step1 = inputs_step1.to(model.device).to(model.dtype)

    text_ids_step1 = model.generate(**inputs_step1, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
    out_text_step1 = processor.batch_decode(text_ids_step1, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print("Step 1 output:", out_text_step1)

    # 将 Step 1 输出加入上下文
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": out_text_step1}]
    })

    # Step 2: 添加音频输入继续推理
    conversation.append({
        "role": "user",
        "content": [
            {"type": "text", "text": cot_user_prompt_step2_no_exp}
            {"type": "audio", "audio": image_item['wav_path']}
        ]
    })

    text_step2 = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios_step2, images_step2, videos_step2 = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs_step2 = processor(
        text=text_step2,
        audio=audios_step2,
        images=images_step2,
        videos=videos_step2,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs_step2 = inputs_step2.to(model.device).to(model.dtype)

    text_ids_step2 = model.generate(**inputs_step2, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
    out_text_step2 = processor.batch_decode(text_ids_step2, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print("Step 2 output (ASR):", out_text_step2)

    # 保存 Step 2 ASR
    with open(output_pred_path, "a", encoding="utf-8") as f:
        try:
            image_item["pred"] = extract_asr_text(out_text_step2)
            f.write(f"{image_item['id']} {image_item['pred']}\n")
        except Exception as e:
            print(f"[WARN] json error: {image_item['id']} {e}")
