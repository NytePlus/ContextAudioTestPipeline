from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

snapshot_path = "/models/models--Vamsi--T5_Paraphrase_Paws/snapshots/f8c3dedd6b6f1bc7db90fee74c5338d6e0f99ba4"
paraphrase_tokenizer = AutoTokenizer.from_pretrained('/models/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1')
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(snapshot_path)

sbert_model = SentenceTransformer('/models/sentence-transformers/all-mpnet-base-v2')

# “一句话是不是语义完整、像人说的” → 通过它生成几个意思相同的句子，然后看它们语义距离有多近。
def semantic_coherence_score(sentence, num_paraphrases=3, device='npu:0'):
    paraphrase_model.to(device)
    
    # 生成 paraphrase
    input_ids = paraphrase_tokenizer(sentence, return_tensors="pt").input_ids.to(device)
    outputs = paraphrase_model.generate(
        input_ids,
        max_length=60,
        num_return_sequences=num_paraphrases,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    paraphrases = [paraphrase_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    
    embeddings = sbert_model.encode([sentence] + paraphrases, convert_to_tensor=True)
    orig_emb = embeddings[0]
    para_embs = embeddings[1:]
    
    cosine_scores = util.cos_sim(orig_emb, para_embs).cpu().numpy().flatten()
    
    score = float(cosine_scores.mean())

    return score, paraphrases

# 1.0
# score, paraphrases = semantic_coherence_score("The cat sat on the mat.")

# 0.97
# print(semantic_coherence_score("And thank you very much good evening everybody and a warm welcome to our next presentation my name is Katharine Morland and together with my colleagues Heitze Robert Young have given me your hand"))
# 0.99
# print(semantic_coherence_score("And thank you very much good evening everybody and a warm welcome to our next presentation My name is Katharina Morlang and together with my colleagues hiker hoods and patrick young please give me your hands")) 

# 0.996
# 0.997 'I coach kids'
# print(semantic_coherence_score("The german sports youth represents all youth sports organizations in germany and we are very happy to be a partner of the follow up project 'I coach kids' class"))
# 0.995
# 0.994
# print(semantic_coherence_score("The German sports youth represents all youth sports organizations in Germany and we are very happy to be a partner of the follow up project 'I coach kids' class"))
# 0.997
# 0.998 'I code it'
# print(semantic_coherence_score("The German sports youth represents all youth sports organizations in Germany and we are very happy to be a partner of the follow up project 'I code it' plus"))

# 0.978
# print(semantic_coherence_score("Martin um if you want and if you are ready you can start now and we will discuss later"))
# 0.978
# print(semantic_coherence_score("Martin um if you want and if you are ready we can start now and we will discuss later"))

# 0.965
# 0.923 'I coach kids'
# print(semantic_coherence_score("Yes and we represent uh the german sports youth and um together we drive the project I coach kids forward germany"))
# 0.972
# 0.985
# print(semantic_coherence_score("And we represent the German sport youths and together we draft the project I coach kids forward in Germany"))
# 0.991
# 0.982
# print(semantic_coherence_score("Yes and we represent the german sports youth and together we drive the project I coach kids forward germany"))