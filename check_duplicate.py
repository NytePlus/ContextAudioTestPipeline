import json
import os
from collections import Counter

def check_duplicate_ids(file_path):
	ids = []
	
	# 1. 读取并提取 ID
	with open(file_path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			
			# 假设 ID 是每行第一个空格前的部分
			# 例如 "YTB+zs9tICC1YuY+00994 And that's..."
			parts = line.split(maxsplit=1)
			if parts:
				ids.append(parts[0])

	# 2. 统计频率
	id_counts = Counter(ids)
	duplicates = {id: count for id, count in id_counts.items() if count > 1}

	# 3. 输出结果
	if not duplicates:
		print("✅ 未发现重复 ID，所有 ID 都是唯一的。")
	else:
		print(f"❌ 发现 {len(duplicates)} 个重复 ID：")
		print("-" * 40)
		for id, count in duplicates.items():
			print(f"ID: {id} | 出现次数: {count}")
		print("-" * 40)
		print(f"总计行数: {len(ids)}")

def inplace_deduplicate_text(file_path):
    seen_ids = set()
    unique_lines = []
    
    # 1. 读入内存并去重
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            uttid = line.split(maxsplit=1)[0]
            if uttid not in seen_ids:
                unique_lines.append(line)
                seen_ids.add(uttid)
    
    # 2. 覆盖写入原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')
            
    print(f"✅ 原文件已更新：删除了 {len(seen_ids) - len(unique_lines)} 条重复记录。")

def compare_ids(raw_jsonl, pred_txt):
    # 1. 提取原始 ID (从 JSONL 的 "key" 字段)
    raw_ids = set()
    with open(raw_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_ids.add(json.loads(line)['key'])

    # 2. 提取预测 ID (从文本行首)
    pred_ids = set()
    with open(pred_txt, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # 取得空格前的第一个部分
                pred_ids.add(line.split(maxsplit=1)[0])

    # 3. 计算差异
    missing_ids = raw_ids - pred_ids    # 原始有，但预测没有 -> 漏跑了
    extra_ids = pred_ids - raw_ids      # 预测有，但原始没有 -> 多跑了或ID写错
    
    # 4. 打印报告
    print(f"📊 比对统计：")
    print(f"原始 ID 总数: {len(raw_ids)}")
    print(f"预测 ID 总数: {len(pred_ids)}")
    print("-" * 30)

    if not missing_ids and not extra_ids:
        print("✅ 完美对齐！ID 完全一致。")
    else:
        if missing_ids:
            print(f"❌ 漏跑 (Missing) {len(missing_ids)} 个 ID:")
            # 打印前 5 个作为示例
            for idx, i in enumerate(list(missing_ids)[:5]):
                print(f"  - {i}")
            if len(missing_ids) > 5: print("    ...")

        if extra_ids:
            print(f"⚠️ 多出 (Extra) {len(extra_ids)} 个 ID:")
            for idx, i in enumerate(list(extra_ids)[:5]):
                print(f"  - {i}")
            if len(extra_ids) > 5: print("    ...")

def remove_ids_from_txt(jsonl_file, txt_file):
    # 1. 提取待删除的 ID 集合 (从 JSONL 的 "key" 字段)
    ids_to_remove = set()
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'key' in data:
                        ids_to_remove.add(data['key'])
                except json.JSONDecodeError:
                    continue
    
    print(f"📌 从 JSONL 中读取了 {len(ids_to_remove)} 个待删除 ID。")

    # 2. 读取 TXT 并过滤内容
    remaining_lines = []
    removed_count = 0
    
    if not os.path.exists(txt_file):
        print(f"❌ 错误: 找不到文件 {txt_file}")
        return

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            # 提取 TXT 行首的 ID (空格前第一个部分)
            parts = stripped_line.split(maxsplit=1)
            uttid = parts[0]
            
            if uttid in ids_to_remove:
                removed_count += 1
            else:
                remaining_lines.append(line) # 保留原始行（含换行符）

    # 3. 写回原 TXT 文件 (原位修改)
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.writelines(remaining_lines)

    print(f"✅ 处理完成！")
    print(f"🗑️ 已从 TXT 中删除行数: {removed_count}")
    print(f"📝 TXT 剩余行数: {len(remaining_lines)}")

# check_duplicate_ids('qwen_asr_train')
# inplace_deduplicate_text('qwen_asr_train')
# compare_ids('/data/slidespeech/train_95/slides/multitask.jsonl', 'qwen_asr_train')
remove_ids_from_txt('/data/slidespeech/test_oracle_v1/slides/multitask.jsonl', 'qwen_asr_train')