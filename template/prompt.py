no_prompt = '{}'

# gt: 同时也能提高肺部的通气功能和氧气的利用效率，长期坚持有氧运动可以降低心血管疾病的风险，改善血脂水平，控制体重，增强免疫系统功能。
default_qwen_sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# prompt='Transcribe the spoken Chinese text, using the provided slide content as reference. Only output the transcription, no extra output.'

# prompt = """
# You are a multimodal speech recognition system.
# Given the input audio and image, perform automatic speech recognition (ASR) on the audio.
# Output only the transcribed spoken text from the audio.
# Do NOT include any OCR results from the image.
# Do NOT describe or reference the image.
# Do NOT add explanations, annotations, or any additional text.
# Return only the final speech transcription.
# """
# 不同类型运动对健康影响。有氧运动，如快走、慢跑、游泳、骑自行车等。力量训练，如举重、使用健身器械、做俯卧撑等。柔韧性训练，如瑜伽、普拉提、拉伸运动等。平衡性训练，如太极、瑜伽中的某些动作、单脚站立等。同时也能提高肺部的通气功能和氧气的利用效率，长期坚持有氧运动可以降低心血管疾病的风险，改善血脂水平，控制体重，增强免疫系统功能。

# prompt = """
# You are an automatic speech recognition (ASR) system.
# Transcribe the speech in the provided audio input.
# Ignore the image entirely.
# Output only the spoken words exactly as heard.
# Do not include OCR content, descriptions, or any extra text.
# """
# 有氧运动能提高肺部通气功能和氧气利用效率，长期坚持能降低心血管疾病风险，改善血脂水平，控制体重，增强免疫系统功能。

# prompt = """
# Your task is automatic speech recognition.
# Use only the audio input to produce a transcription.
# The image input must be ignored completely.
# Return only the recognized speech text, with no explanations, comments, or additional information.
# """
# 不同类型运动对健康影响。有氧运动，如快走、慢跑、游泳、骑自行车等。力量训练，如举重、使用健身器械、做俯卧撑等。柔韧性训练，如瑜伽、普拉提、拉伸运动等。平衡性训练，如太极、瑜伽中的某些动作、单脚站立等。同时也能提高肺部的通气功能和氧气的利用效率。长期坚持有氧运动可以降低心血管疾病的风险，改善血脂水平，控制体重，增强免疫系统功能。

# prompt = """
# You are a multimodal perception system.
# Use the image to extract OCR text.
# Use the audio to perform automatic speech recognition (ASR).
# Return the result strictly in the following JSON format:

# {
#   "ocr_text": "<text recognized from the image>",
#   "asr_text": "<speech transcribed from the audio>"
# }

# Do not add any extra fields.
# Do not include explanations or additional text outside the JSON.
# """
# {
#    "ocr_text": "不同类型运动对健康影响",
#    "asr_text": "同时也能提高肺部的通气功能和氧气的利用效率长期坚持有氧运动可以降低心血管疾病的风险改善血脂水平控制体重增强免疫系统功能"
# }


ocr_asr_format_prompt = """
Perform automatic speech recognition (ASR) on the audio.
When transcribing the speech, you may use the image content
as contextual and semantic guidance to improve recognition accuracy,
such as resolving ambiguities, terminology, or topic relevance.

Do NOT copy image content directly into the ASR output
unless it is actually spoken in the audio.

Return the result strictly in the following JSON format:

{
  "ocr_text": "<text recognized from the image, if any>",
  "asr_text": "<speech transcribed from the audio>"
}

Do not add any extra fields.
Do not include explanations or text outside the JSON.
"""
"""
可能出现
1. 以为自己不能asr:  I'm sorry, I can't perform ASR directly. But if you can try some ASR tools like Google Speech-to-Text or IBM Watson Speech to Text. They can help you with the transcription. If you have any other other questions or need more help, feel free to let me know.
2. 被OCR的文字干扰，输出重复乱码: I Human: What I What's the digital divide?
3.  "You Human:"标签
"""

# prompt = """
# Perform automatic speech recognition (ASR) on the audio.
# When transcribing the speech, you may use the image content
# as contextual and semantic guidance to improve recognition accuracy,
# such as resolving ambiguities, terminology, or topic relevance.

# Do NOT copy image content directly into the ASR output
# unless it is actually spoken in the audio.

# Return the result strictly in the following JSON format:

# {
#   "asr_text": "<speech transcribed from the audio>"
# }

# Do not add any extra fields.
# Do not include explanations or text outside the JSON.
# """

asr_imgref_format_prompt = """
You are an automatic speech recognition (ASR) model.
Your task is to transcribe the given audio.

Use image content only as contextual guidance to improve transcription accuracy.
Do NOT invent any content from the images; transcribe only what is spoken.

Strictly return the result as a single JSON object, in the exact format below:
No additional text, no explanations, no extra fields, no "Human:" labels.

{
  "asr_text": "<transcribed speech here>"
}

If you cannot transcribe, output an empty string:
{
  "asr_text": ""
}
"""

# hotword_prompt = """
# Transcribe speech to text. Use keywords to improve speech recognition accuracy. But if the keywords are irrelevant, just ignore them. The keywords are 
# {}
# # Do NOT add explanations, annotations, or any additional text.
# # Return only the final speech transcription.
# # IF you cannot transcribe it, return nothing.
# """

hotword_prompt = """
Transcribe speech to text. Use keywords to improve speech recognition accuracy. The keywords are 
{}
"""

hotword_format_prompt = """
Transcribe speech to text. Use keywords to improve speech recognition accuracy. The keywords are 
{}
# Do NOT add explanations, annotations, or any additional text.
# Return only the final speech transcription.
Return the result strictly in the following JSON format:
{{
  "asr_text": "<speech transcribed from the audio>"
}}
If you cannot transcribe, output an empty string:
{{
  "asr_text": ""
}}
"""
"""
加入关键词会降低效果，出现包括：
1. 会降低语音的关注度：I'm sorry, but I can't transcribe speech to text without actually hearing the speech.
2. 复述关键词：Antioppressive classroom, Diversity students, Impact students, Students analyze, Classroom selecta, Students privileged, Culturally students, Diversity amongs, Students responsive, Tan antioppressive, Amongs students, Antioppression approaches, Analyze privilege, Students 12, Teaching focusing, Choice approaches, Select apply, Know teaching, Way marginalized, Analyze hhw, Analyze position, 29 analyze, Use support, Support tan, Counts choice, Impacte, Sani impacted, Approaches sandorn, Talk, Affirm, Responsive way, Use, Materials use, Acknowle, Andori materials, 47 cc, Senee total, Acknowledge, Ffocus tall, Hhw, Impacted ton, Culttrally, Ledge, Know, Tton, 16, 100, Way, 41 06.
"""

# asr_prompt = """
# Transcribe speech to text. Do NOT add explanations, annotations, or any additional text. 
# """

# asr_prompt = """
# Transcribe speech to text. Do NOT add explanations, annotations, or any additional text. 
# Return the result strictly in the following JSON format:
# {
#   "asr_text": "<speech transcribed from the audio>"
# }
# """

asr_format_prompt = """
You are an automatic speech recognition (ASR) model.
Your task is to transcribe the given audio.

Transcribe speech to text. Do NOT add explanations, annotations, or any additional text. 
Return the result strictly in the following JSON format:
{
  "asr_text": "<speech transcribed from the audio>"
}
If you cannot transcribe, output an empty string:
{
  "asr_text": ""
}
"""
"""
可能会出现：
1. 音频非常短时（比如uh，okay, next one please），转录文本会生成很长一段不相干文字
"""

cot_sys_prompt = """
You are a multimodal assistant with vision and audio understanding.

Input:
- A list of keywords
- One image
- One audio clip

Perform TWO steps and output JSON only.

Step 1: Image-based keyword explanation  
For each keyword:
- Use the image if relevant; otherwise use general knowledge and mark as not visible.
- Explanation must be NO MORE THAN 15 characters.
- Do NOT use audio.

Output:
{
  "<keyword>": "<no more than 15 characters>",
  ...
}

Step 2: Keyword-aware speech transcription  
Transcribe the audio using the keywords and Step 1 as context.

Output:
{
  "asr_text": "<audio transcription>"
}

Constraints:
- Exactly TWO JSON blocks.
- No extra text.
"""

cot_user_prompt = """
Keywords: {}
"""

cot_user_prompt_step1 = """
You are a multimodal assistant with vision understanding.

Input:
- A list of keywords
- One image

Task:
- For each keyword, provide a short explanation (max 15 characters)
- Use the image if relevant; otherwise mark as not visible
- Do NOT use audio

Output:
{{
  "<keyword>": "<no more than 15 characters>",
  ...
}}

Constraints:
- Output JSON only
- Do not add any extra text

Keywords: {}
"""

cot_user_prompt_step2 = """
You are a multimodal assistant with audio understanding.

Input:
- The keyword explanations from Step 1
- One audio clip

Task:
- Transcribe the audio using the keywords as context

Output:
{{
  "asr_text": "<audio transcription>"
}}

Constraints:
- Output JSON only
- Do not hallucinate

Keyword explanations: {}
"""

cot_user_prompt_step2_no_exp = """
You are a multimodal assistant with audio understanding.

Input:
- The keyword explanations from Step 1
- One audio clip

Task:
- Transcribe the audio using the keywords as context

Output:
{{
  "asr_text": "<audio transcription>"
}}

Constraints:
- Output JSON only
- Do not hallucinate
"""

asr_prompt = """
You are an automatic speech recognition (ASR) model.
Your task is to transcribe the given audio.

Transcribe speech to text. Do NOT add explanations, annotations, or any additional text. 
Only output the transcription.
"""

asr_format_prompt = """
You are an automatic speech recognition (ASR) model.
Your task is to transcribe the given audio.

Only output the transcription.
Strictly return the result as a single JSON object, in the exact format below:

{
  "asr_text": "<transcribed speech here>"
}
"""

qwen_asr_prompt = """
Detect the language and recognize the speech: <|en|>
"""