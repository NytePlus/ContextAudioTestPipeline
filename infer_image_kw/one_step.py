import json
import os
import re
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from data import Data
from template.prompt import cot_sys_prompt, cot_user_prompt
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
# 批量推理
# ---------------------------
for image_item, kw_item in tqdm(zip(data, kw), desc="推理中"):
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": cot_sys_prompt}
            ],
        },
        {
            "role": "user",
            "content": [],
        },
    ]

    kws = hotword_process(kw_item["hotword"])
    prompt = cot_user_prompt.format(kws)
    conversation[1]["content"].append({"type": "text", "text": prompt})

    conversation[1]["content"].append({"type": "image", "image": image_item['ppt_path']})
    conversation[1]["content"].append({"type": "audio", "audio": image_item['wav_path']})

    # 准备推理输入
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # 推理生成
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)

    # 解析文本
    out_text = processor.batch_decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print(out_text)
    continue

    with open(output_pred_path, "a", encoding="utf-8") as f:
        item = image_item
        parts = out_text.split("\nassistant\n", 1)
        if len(parts) == 2:
            try:
                pred_text = parts[1]
                item["pred"] = extract_asr_text(pred_text)
                f.write(f"{item['id']} {item['pred']}\n")
            except:
                print(f"[WARN] json error: {item['id']} {pred_text}")
        else:
            print(f"[WARN] parse error: {item['id']}")


