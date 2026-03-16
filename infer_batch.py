from tqdm import tqdm
from data import Data, Out
from pipelines.qwen_asr import QwenASRPipeline
from template.ans_extract import extract_asr_from_pred
from template.prompt import asr_format_prompt, asr_prompt

data_dir = '/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech'
input_json = "test_oracle_v1/slides/multitask.jsonl"
output_pred = "qwen_asr_train"
batch_size = 12

data = Data(input_json, data_dir, audio_format='ark')
out = Out(output_pred)
pipe = QwenASRPipeline()

FORMAT = False

batch_ids, batch_convs = [], []
all_ids, all_preds = [], []
ids, preds = [], []

for item in tqdm(data, desc="推理中"):
    if out.is_processed(item['id']):
        continue

    conversation = pipe.chat_template(asr_format_prompt if FORMAT else asr_prompt, item['audio_path'])

    batch_convs.append(conversation)
    batch_ids.append(item['id'])

    if len(batch_convs) == batch_size:
        batch_preds = pipe.run_batch(batch_convs)
        all_ids.extend(batch_ids)
        all_preds.extend(batch_preds)
        batch_convs, batch_ids = [], []

if batch_convs:
    batch_preds = pipe.run_batch(batch_convs)
    all_ids.extend(batch_ids)
    all_preds.extend(batch_preds)

for id, pred in zip(all_ids, all_preds):
    try:
        asr = extract_asr_from_pred(pred) if FORMAT else pred
        ids.append(id)
        preds.append(asr)
    except:
        print(f'{id} fail')

out.append(ids, preds)
