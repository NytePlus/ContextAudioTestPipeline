import re
import json

def extract_asr_from_raw(raw_text: str) -> str:
    parts = raw_text.split("\nassistant\n", 1)
    if len(parts) == 2:
            pred_text = parts[1]
            return extract_asr_from_pred(pred_text)
    else:
        raise ValueError(f"assistant parse error: {raw_text}")
    
def extract_asr_from_pred(pred_text):
    try:
        match = re.search(r'\{[\s\S]*?\}', pred_text)
        if not match:
            raise ValueError("No JSON object found")

        json_str = match.group(0).replace('\n', ' ')
        data = json.loads(json_str)
        return data["asr_text"].replace('\n', ' ')
    except:
        raise ValueError(f"json error: {pred_text}")

def extract_asr_from_think(pred_text):
    pred_text = re.sub(r'<think>.*?</think>', '', pred_text, flags=re.DOTALL)
    pred_text = re.sub(r'[^a-zA-Z0-9\s]', '', pred_text).lower().strip()
    return pred_text