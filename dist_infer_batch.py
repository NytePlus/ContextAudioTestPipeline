# torchrun --nproc_per_node=8 --master_port=29500 dist_infer_batch.py
import os
import glob
import shutil
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from data import Data, Out
from pipelines.qwen_asr import QwenASRPipeline
from template.ans_extract import extract_asr_from_pred
from template.prompt import asr_format_prompt, asr_prompt

data_dir = '/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech'
input_json = "train_95/slides/multitask.jsonl"
output_pred = 'qwen_asr_train'
batch_size = 256
FORMAT = False

def process_and_save(pipe, out, ids, convs, format_flag):
    batch_preds = pipe.run_batch(convs)
    final_ids, final_preds = [], []
    
    for id, pred in zip(ids, batch_preds):
        try:
            asr = extract_asr_from_pred(pred) if format_flag else pred
            final_ids.append(id)
            final_preds.append(asr)
        except:
            print(f'Rank {dist.get_rank()}: {id} extraction failed')
    
    out.append(final_ids, final_preds)

if __name__ == "__main__":
    if "RANK" in os.environ:
        dist.init_process_group("hccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = f"npu:{local_rank}" # 自动绑定当前进程的 NPU
    torch.npu.set_device(device)

    # 2. 数据集切分
    # dataset = Data(input_json, data_dir, audio_format='base64')
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # dataloader = DataLoader(
    #     dataset, 
    #     batch_size=batch_size, 
    #     sampler=sampler,
    #     num_workers=16,
    #     collate_fn=lambda x: x
    # )

    # rank_pred = f"{output_pred}_rank{rank}"
    # out = Out(rank_pred)
    
    # # 3. 加载模型到指定的 local_rank
    # pipe = QwenASRPipeline(device=device) 
    # pbar = tqdm(dataloader, desc=f"Rank {rank} 推理中") if rank == 0 else dataloader

    # for batch in pbar:
    #     batch_ids = [item['id'] for item in batch]
    #     batch_audios = [item['audio_path'] for item in batch]
        
    #     batch_convs = [
    #         pipe.chat_template(asr_format_prompt if FORMAT else asr_prompt, audio_path)
    #         for audio_path in batch_audios
    #     ]
        
    #     process_and_save(pipe, out, batch_ids, batch_convs, FORMAT)

    if dist.is_initialized():
        dist.barrier()

    # 5. 仅由 Rank 0 进行合并操作
    if rank == 0:
        temp_files = glob.glob(f"{output_pred}_rank*")
        temp_files.sort()

        with open(output_pred, 'w', encoding='utf-8') as outfile:
            for temp_file in tqdm(temp_files, desc=f"Rank {rank} 正在合并结果"):
                with open(temp_file, 'r', encoding='utf-8') as infile:
                    shutil.copyfileobj(infile, outfile)

                os.remove(temp_file)

        print(f"所有结果已汇总至: {output_pred}")

    # 6. 销毁进程组
    if dist.is_initialized():
        dist.destroy_process_group()