import soundfile as sf
from pipelines.LongCatFlashOmni.longcat_omni_for_outter_import import LoncatOmniInfer

class LongcatOmni:
    def __init__(self, model_path="/models/meituan-longcat/LongCat-Flash-Omni"):
        from argparse import Namespace

        args = Namespace(
            model_path=model_path,
            output_dir="./output",
            nodes=None,
            node_rank=0,
            dist_init_addr=None,
            tp_size=8,
            ep_size=8,
            encoder_device="cuda",
        )

        self.infer_engine = LoncatOmniInfer(args)

        self.sampling_params = {
            "temperature": 1.0,
            "max_new_tokens": 4096,
            "top_p": 1.0,
            "top_k": 1,
            "repetition_penalty": 1.0,
            "ignore_eos": True,
        }

    def chat_template(self, prompt, ppt_path, wav_path):
        """
        构造 omni 输入（这是给 run_batch 用的）
        """
        return {
            "prompt": prompt,
            "ppt_path": ppt_path,
            "wav_path": wav_path,
        }

    def run_batch(self, conversations):
        """
        conversations: List[Dict]
        """
        results = []

        for conv in conversations:
            omni_input = self._build_omni_input(conv)
            result = self.infer_engine.generate(
                omni_input,
                sampling_params=self.sampling_params,
            )
            results.append(result)

        return results

    def _build_omni_input(self, conv):
        """
        转成 LoncatOmniInfer 能吃的 input dict
        """
        # 1. 读音频
        audio, sr = sf.read(conv["wav_path"])
        assert sr == 16000

        # 2. 构造 omni 输入
        input_dict = {
            "prompt": conv["prompt"],      # 文本 prompt
            "images": [conv["ppt_path"]],  # 支持多张
            "audio": audio,
        }

        return input_dict
