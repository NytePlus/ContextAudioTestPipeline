import base64
import re
import json
import io
import os
from pathlib import Path
from tqdm import tqdm

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from research_qwen.embed_simi import embed_sim
from research_qwen.res_utils import find_generated_spans, find_audio_span
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


data = Data(INPUT_JSONL, DATA_DIR)
processor = AutoProcessor.from_pretrained(MODEL_NAME, sampling_rate=16000)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="npu:0",
    attn_implementation="eager"
).eval()


def load_audio_from_data_uri(data_uri, sr=16000):
    if data_uri.startswith("data:audio"):
        base64_str = data_uri.split(",")[1]
    else:
        base64_str = data_uri
    audio_bytes = base64.b64decode(base64_str)
    audio_file = io.BytesIO(audio_bytes)
    waveform, _ = librosa.load(audio_file, sr=sr)
    return waveform


def labels_from_inputs(input_ids):
    labels = input_ids.clone()

    # 默认全部不计 loss
    labels[:] = -100

    # 对每个样本，把 gt 对应的 token 打开 loss
    for i, gt in enumerate([item['gt']]):
        gt_ids = processor(
            gt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids[0].to(input_ids.device)

        # 找到 gt 在 input_ids 里的位置（假设 gt 在最后）
        labels[i, -len(gt_ids):] = gt_ids

    return labels


if __name__ == "__main__":
    i, n = 0, 10
    for item in tqdm(data):
        if i > n: break
        audios = [load_audio_from_data_uri(item["wav_path"])]
        texts = ["<|audio_bos|><|AUDIO|><|audio_eos|>" + qwen_asr_prompt + item['gt']]

        inputs = processor(
            text=texts,
            audios=audios,
            return_tensors="pt",
            sampling_rate=16000,
            padding=True
        ).to(model.device)
        
        outputs = model(
            **inputs, 
            output_attentions=True,
            output_hidden_states=True,
        )

        attn_map_list = outputs.attentions # torch.Size([1, 32, 357, 357])
        hiddens = outputs.hidden_states # torch.Size([1, 357, 4096])

        token_texts = [[processor.decode([tok], skip_special_tokens=False) for tok in input_ids]
                       for input_ids in inputs['input_ids']]
        speech_token_id = processor.audio_token_id
        audio_starts, audio_lengths = find_audio_span(inputs["input_ids"], speech_token_id)

        labels = labels_from_inputs(inputs['input_ids'])
        gen_starts, gen_lengths = find_generated_spans(inputs['input_ids'], labels)

        attn_output = os.path.join('research_qwen', 'attention_maps')
        sim_output = os.path.join('research_qwen', 'coarse sim')
        norm_output = os.path.join('research_qwen', 'hidden_norm')
        os.makedirs(attn_output, exist_ok=True)
        os.makedirs(sim_output, exist_ok=True)
        os.makedirs(norm_output, exist_ok=True)

        embed_sim([item['id']], attn_map_list, hiddens, token_texts, gen_starts, gen_lengths, audio_starts, audio_lengths, attn_output, sim_output, norm_output)