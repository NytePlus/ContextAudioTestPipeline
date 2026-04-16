import json
import os
import base64
import kaldiio
import soundfile as sf

from pathlib import Path
from io import BytesIO


def ark_to_base64_audio(ark_path: str, audio_format="wav") -> str:
    """
    Convert kaldi ark:offset audio to base64 data URI.
    Returns: data:audio/wav;base64,...
    """
    # 1. 读取 ark 音频
    sr, wav = kaldiio.load_mat(ark_path)

    # 2. 写入内存 buffer
    buffer = BytesIO()
    sf.write(buffer, wav, sr, format=audio_format.upper())
    buffer.seek(0)

    # 3. base64 编码
    audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    # 4. 组装 data URI
    return f"data:audio/{audio_format};base64,{audio_base64}"

class Data:
    def __init__(self, name, root_dir, audio_format, part=None):
        self.root_dir = root_dir
        path = Path(root_dir) / name
        self.audio_format = audio_format

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        elif path.suffix == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            self.data=data 
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        self.format_data = []
        for item in self.data[:part]:
            id = self.get_id(item)
            ppt_path = self.get_pptpath(item)
            audio_path = self.get_audiopath(item, audio_format)
            context_type = 'none'
            if item.get('history', ''):
                context = item.get('history', '')
                context_type = 'history'
            else:
                context = item.get('hotword', '')
                context_type = 'hotword'
            gt = self.get_gt(item)
            
            self.format_data.append({
                'id': id,
                "ppt_path": ppt_path,
                "audio_path": audio_path,
                "context": context,
                "gt": gt,
                "context_type": context_type
            })

    def get_id(self, item):
        if "key" in item:
            return item['key']
        return ''

    def get_pptpath(self, item):
        if "ppt_path" in item:
            return os.path.join(self.root_dir, item["ppt_path"])
        elif "image" in item:
            return item['image']
        return ''

    def get_audiopath(self, item, audio_format):
        if audio_format == 'wav':
            assert "wav_path" in item
            return os.path.join(self.root_dir, item["wav_path"])
        elif audio_format == 'ark':
            assert "path" in item
            return item['path']
        elif audio_format == 'base64':
            assert "path" in item
            return item['path'] # 在加载时编码
        elif audio_format == 'url':
            assert "path" in item
            return item['path']
        return ''
    
    def get_gt(self, item):
        if "gt_text" in item:
            return item['gt_text']
        elif "target" in item:
            return item['target']
        raise ValueError(item)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for item in self.format_data:
            audio_path = item['audio_path']
            if self.audio_format == 'ark':
                audio_path = ark_to_base64_audio(audio_path)
            item['audio_path'] = audio_path
            yield item
    
    def __getitem__(self, idx):
        item = self.format_data[idx]
        wav_path = item['audio_path']
        if self.audio_format == 'ark':
            wav_path = ark_to_base64_audio(wav_path)
        item['audio_path'] = wav_path
        return item

    def dump_pred(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.format_data:
                line = f"{item['id']} {item['pred']}\n"
                f.write(line)

    def dump_json(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.format_data, f, ensure_ascii=False, indent=4)

class Out:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.processed_ids = set()
        if self.output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        processed_id = line.split()[0]
                        self.processed_ids.add(processed_id)

    def is_processed(self, id):
        return str(id) in self.processed_ids
    
    def append(self, ids, preds):
        with open(self.output_path, "a", encoding="utf-8") as f:
            for id, pred in zip(ids, preds):
                f.write(f"{id} {pred}\n")