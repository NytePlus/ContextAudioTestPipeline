from tqdm import tqdm
from data import Data, Out
from template.prompt import prompt
from template.ans_extract import extract_asr_from_pred
from pipelines.ming_omni import MingOmni

data_dir = '/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech'
input_json = "test_oracle_v1/slides/multitask.jsonl"
output_pred = "output_pred"
batch_size = 1

data = Data(input_json, data_dir)
out = Out(output_pred)
pipe = MingOmni()

batch_ids, batch_convs = [], []
all_ids, all_preds = [], []
ids, preds = [], []

for item in tqdm(data, desc="推理中"):
    if out.is_processed(item['id']):
        continue

    conversation = pipe.chat_template(prompt, item['ppt_path'], item['audio_path'])

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
        asr = extract_asr_from_pred(pred)
        ids.append(id)
        preds.append(asr)
    except:
        print(f'{id} fail')

out.append(ids, preds)
