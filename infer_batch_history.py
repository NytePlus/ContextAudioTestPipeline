from tqdm import tqdm
from data import Data, Out
from template.prompt import no_prompt
from pipelines.doubao import DoubaoPipeline
from template.ans_extract import extract_asr_from_think

data_dir = '/hpc_stor03/sjtu_home/chencheng.wang/data/canvas'
input_json = "multitask.jsonl"
output_pred = "output_pred"
batch_size = 8

data = Data(input_json, data_dir, audio_format='url')
out = Out(output_pred)
pipe = DoubaoPipeline()

batch_ids, batch_convs = [], []
all_ids, all_preds = [], []
ids, preds = [], []

for item in tqdm(data, desc="推理中"):
    if out.is_processed(item['id']):
        continue

    prompt = no_prompt
    prompt = prompt.format(item['context'])
    conversation = pipe.chat_template(prompt, item['audio_path'])

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
        asr = pred
        ids.append(id)
        preds.append(asr)
    except:
        print(f'{id} fail')

out.append(ids, preds)
