import os
import torch
import torch.nn.functional as F

from plot import save_heatmap, save_scalar, save_scalar_withstr
from res_utils import find_generated_spans, decode_replace_audios

def outputs_from_aispeechasr(encode_feature_length, batch, model_outputs, model):
    audio_lengths = encode_feature_length.detach().cpu().numpy().tolist()
    input_ids = batch['input_ids']
    token_texts = [
        decode_replace_audios(token_ids, model.tokenizer, audio_length)
        for token_ids, audio_length in zip(input_ids, audio_lengths)
    ]
    attn_map_list = model_outputs.attentions # torch.Size([1, 32, 357, 357])

    hiddens = model_outputs.hidden_states # torch.Size([1, 357, 4096])

    speech_token_id = tokenizer.default_speech_token

    labels = batch['labels']

    batch_idx, audio_starts = (input_ids == speech_token_id).nonzero(as_tuple=True)

    gen_starts, gen_lengths = find_generated_spans(input_ids, labels)


def embed_sim(keys, attn_map_list, hiddens, 
              token_texts, gen_starts, gen_lengths, audio_starts, audio_lengths,
              attn_output, sim_output, norm_output, 
              ):
    # --- attention map ---
    for layer, attn_maps in enumerate(attn_map_list):
        if attn_maps is None:
            continue
        for i, (key, attn_map, token_text) in enumerate(zip(keys, attn_maps, token_texts)):
            save_path=os.path.join(attn_output, f'attn_{key}_{layer}.png')
            if os.path.exists(save_path):
                continue
            save_heatmap(
                matrix=attn_map[0].cpu().detach().to(torch.float32).numpy(), 
                save_path=save_path,
                x_labels=token_text,
                y_labels=token_text
            )
            print(f'save to attn_{key}_{layer}.png')
            
    # --- 粗粒度跨模态相似度 ---

    cos_sims, enc_dist = {}, {}
    for layer, hidden in enumerate(hiddens):
        for key, audio_start, audio_length, gen_start, gen_length in zip(keys, audio_starts, audio_lengths, gen_starts, gen_lengths):
            h1_mean, h2_mean, cos_sim, euc_dist = compute_coarse_similarity(hidden, audio_start, audio_length, gen_start, gen_length)
            if cos_sims.get(key) is None:
                cos_sims[key] = []
            cos_sims[key].append(cos_sim)
            if enc_dist.get(key) is None:
                enc_dist[key] = []
            enc_dist[key].append(euc_dist)
    for key in keys:
        save_path = os.path.join(sim_output, key)
        if not os.path.exists(save_path):
            save_scalar(save_path, cos_sims[key], "cosine similarity")
            save_scalar(save_path, enc_dist[key], "enc distance")

    # --- 嵌入空间大小 ---
    for layer, hidden in enumerate(hiddens):
        l2_norms = torch.norm(hidden[0], p=2, dim=-1).detach().cpu().tolist()

        for b, key in enumerate(keys):
            x_labels = token_texts[b]

            assert len(x_labels) == len(l2_norms), \
                f"Length mismatch: tokens={len(x_labels)}, norms={len(l2_norms)}"

            save_path = os.path.join(norm_output, f"norm_{key}_{layer}.png")
            if os.path.exists(save_path):
                continue
            save_scalar_withstr(
                x_labels=x_labels,
                y_values=l2_norms,
                save_path=save_path,
                xlabel="Token",
                ylabel="L2 Norm",
                title=f"Layer {layer} | Key {key} | Hidden L2 Norm"
            )

def compute_coarse_similarity(H, audio_start, audio_length, gen_start, gen_length):
    """
    H: [1, seq_len, hidden_dim] tensor
    audio_start, audio_length: int
    gen_start, gen_length: int

    Returns:
        h1_mean: [hidden_dim] tensor
        h2_mean: [hidden_dim] tensor
        cosine_sim: float
        euclidean_dist: float
    """
    # 取 slice
    h1 = H[:, audio_start:audio_start+audio_length, :]   # [1, audio_length, hidden_dim]
    h2 = H[:, gen_start:gen_start+gen_length, :]         # [1, gen_length, hidden_dim]

    # 沿 seq_len 维求均值
    h1_mean = h1.mean(dim=1).squeeze(0)  # [hidden_dim]
    h2_mean = h2.mean(dim=1).squeeze(0)  # [hidden_dim]

    # 余弦相似度
    cosine_sim = F.cosine_similarity(h1_mean.unsqueeze(0), h2_mean.unsqueeze(0)).item()

    # 欧几里得距离
    euclidean_dist = torch.norm(h1_mean - h2_mean, p=2).item()

    return h1_mean, h2_mean, cosine_sim, euclidean_dist