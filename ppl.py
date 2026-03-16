import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "/models/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="npu:0"
)
model.eval()

def ppl(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=False
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    ppl = math.exp(loss.item())
    return ppl

# 47.09
# print(ppl("This is a simple example sentence to test perplexity."))

# 59.62
# print(ppl("And thank you very much good evening everybody and a warm welcome to our next presentation my name is Katharine Morland and together with my colleagues Heitze Robert Young have given me your hand"))
# 50.45
# print(ppl("And thank you very much good evening everybody and a warm welcome to our next presentation My name is Katharina Morlang and together with my colleagues hiker hoods and patrick young please give me your hands")) 

# 对German、german大小写很敏感
# 132.31
# 106.77 'I coach kids'
# print(ppl("The german sports youth represents all youth sports organizations in germany and we are very happy to be a partner of the follow up project I coach kids class"))
# 101.85
# 85.90
# print(ppl("The German sports youth represents all youth sports organizations in Germany and we are very happy to be a partner of the follow up project I coach kids class"))
# 107.17
# 86.95 'I code it'
# print(ppl("The German sports youth represents all youth sports organizations in Germany and we are very happy to be a partner of the follow up project I code it plus"))

# 50.84
# print(ppl("Martin um if you want and if you are ready you can start now and we will discuss later"))
# 58.56
# print(ppl("Martin um if you want and if you are ready we can start now and we will discuss later"))

# 对专业术语打引号很敏感
# 1137.97
# 578.93 'I coach kids'
# print(ppl("Yes and we represent uh the german sports youth and um together we drive the project I coach kids forward germany"))
# 646.30
# 329.78
# print(ppl("And we represent the German sport youths and together we draft the project I coach kids forward in Germany"))
# 309.18
# print(ppl("Yes and we represent the german sports youth and together we drive the project 'I coach kids' forward germany"))