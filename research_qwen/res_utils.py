import torch

def decode_replace_audios(token_ids, tokenizer, audio_length):
    decoded = []

    for tok in token_ids:
        if tok == tokenizer.default_speech_token:
            decoded.extend(['<AUDIO>'] * audio_length)
        else:
            decoded.append(tokenizer.decode([tok], skip_special_tokens=False))
    return decoded

def find_generated_spans(input_ids, labels):
    B, T = input_ids.shape
    starts, lengths = [], []

    for b in range(B):
        valid_pos = (labels[b] != -100).nonzero(as_tuple=True)[0]

        if valid_pos.numel() == 0:
            starts.append(None)
            lengths.append(0)
            continue

        start_idx = valid_pos[0].item()
        end_idx   = valid_pos[-1].item() + 1
        length    = end_idx - start_idx

        starts.append(start_idx)
        lengths.append(length)

    return starts, lengths

def find_audio_span(input_ids, speech_token_id):
    B, L = input_ids.shape
    audio_starts = torch.zeros(B, dtype=torch.long)
    audio_lengths = torch.zeros(B, dtype=torch.long)

    for b in range(B):
        seq = input_ids[b]

        # 找到所有 speech token 的位置
        idx = (seq == speech_token_id).nonzero(as_tuple=False).squeeze(-1)

        if idx.numel() == 0:
            raise ValueError(f"No speech token found in sample {b}")

        # 起始位置
        start = idx[0].item()

        # 连续长度（保证连续）
        # idx 应该是 [k, k+1, k+2, ...]
        length = idx.numel()

        audio_starts[b] = start
        audio_lengths[b] = length

    return audio_starts, audio_lengths