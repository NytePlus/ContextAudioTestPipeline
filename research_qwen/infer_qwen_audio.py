import base64
import re
import json
import io
from pathlib import Path
from tqdm import tqdm

import librosa
import torch
from transformers import AutoProcessor
from research_qwen.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
from data import Data
from template.prompt import asr_text_prompt, qwen_asr_prompt

# ======================
# 配置
# ======================
MODEL_NAME = "/models/Qwen/Qwen2-Audio-7B"
BATCH_SIZE = 1

DATA_DIR = "/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech"
INPUT_JSONL = "test_oracle_v1/multitask.jsonl"
OUTPUT_PATH = Path("qwen_audio_pred2.txt")

# ======================
# 加载模型
# ======================
data = Data(INPUT_JSONL, DATA_DIR)
processor = AutoProcessor.from_pretrained(MODEL_NAME, sampling_rate=16000)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="npu:0"
).eval()


# ======================
# 工具函数
# ======================
def load_audio_from_data_uri(data_uri, sr=16000):
    if data_uri.startswith("data:audio"):
        base64_str = data_uri.split(",")[1]
    else:
        base64_str = data_uri
    audio_bytes = base64.b64decode(base64_str)
    audio_file = io.BytesIO(audio_bytes)
    waveform, _ = librosa.load(audio_file, sr=sr)
    return waveform

def extract_asr_text(text: str) -> str:
    text = text.split('Detect the language and recognize the speech: ')[-1]
    return text.strip()
# qwen-audio容易输出json是单引号没法parse

# ======================
# 构造 batch
# ======================
def build_conversations(items):
    conversations = []
    audios = []

    for item in items:
        conversations.append([
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": item["wav_path"]},
                    {"type": "text", "text": qwen_asr_prompt},
                ],
            }
        ])
        audios.append(load_audio_from_data_uri(item["wav_path"]))

    texts = [
        # processor.apply_chat_template(
        #     conv,
        #     add_generation_prompt=False,
        #     tokenize=False
        # )
        "<|audio_bos|><|AUDIO|><|audio_eos|>" + qwen_asr_prompt
        for conv in conversations
    ]

    return texts, audios


# ======================
# Batch 推理
# ======================
@torch.no_grad()
def run_batch(items):
    texts, audios = build_conversations(items)

    inputs = processor(
        text=texts,
        audios=audios,
        return_tensors="pt",
        sampling_rate=16000,
        padding=True
    ).to(model.device)

    gen_ids = model.generate(
        **inputs,
        max_length=600
    )

    outputs = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    results = []
    for item, out in zip(items, outputs):
        pred = extract_asr_text(out)
        print(pred)
        results.append((item["id"], pred))

    return results


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_ids.add(line.split()[0])

    batch = []
    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
        for item in tqdm(data):
            if item["id"] in processed_ids:
                continue

            batch.append(item)

            if len(batch) == BATCH_SIZE:
                results = run_batch(batch)
                for uid, pred in results:
                    fout.write(f"{uid} {pred}\n")
                batch = []

        if batch:
            results = run_batch(batch)
            for uid, pred in results:
                fout.write(f"{uid} {pred}\n")
