import asyncio
import time
import uuid
import json
import aiohttp
import concurrent
from typing import Any, Dict, Optional

class ASRClient:
    """字节跳动（火山引擎）大模型录音文件识别客户端"""

    def __init__(
            self,
            app_key: str,
            access_key: str,
            resource_id: str = "volc.seedasr.auc",
    ):
        """
        初始化 ASR 客户端 (使用新版控制台的鉴权方式)
        :param api_key: 火山引擎控制台获取的 API Key
        :param resource_id: 资源ID，默认使用豆包录音文件识别模型2.0 (volc.seedasr.auc)
        """
        self.app_key = app_key
        self.access_key = access_key
        self.resource_id = resource_id
        self.submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
        self.query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"

    async def asr(self,
                  audio_path: str,
                  audio_format: str,
                  context: str,
                  uid: str = "default_user",
                  enable_speaker_info: bool = False,
                  timeout: int = 300,
                  poll_interval: float = 3.0) -> Optional[Dict[str, Any]]:
        """
        提交识别任务并轮询获取结果
        :param audio_url: 音频的公网访问链接
        :param audio_format: 音频格式 (例如: mp3, wav, ogg)
        :param uid: 用户标识 (推荐传 IMEI 或 MAC，这里默认占位)
        :param enable_speaker_info: 是否开启说话人分离
        :param timeout: 轮询超时时间(秒)
        :param poll_interval: 每次轮询的间隔(秒)
        :return: 识别结果字典，包含文本和时间戳分句信息
        """
        # 生成唯一的请求 ID
        task_id = str(uuid.uuid4())

        # ---------------- 1. 提交任务阶段 ----------------
        submit_headers = {
            "X-Api-App-Key": self.app_key,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": task_id,
            "X-Api-Sequence": "-1",
            "Content-Type": "application/json"
        }
        audio_url = audio_path

        context = {
                    "context_type": "dialog_ctx",
                    "context_data":[
                        {"text": context},
                    ]
                }
        context_str = json.dumps(context, indent=4, ensure_ascii=False)

        submit_payload = {
            "user": {
                "uid": uid
            },
            "audio": {
                "url": audio_url,
                "format": audio_format,
                "context": context_str
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,  # 启用文本规范化 (123美元)
                "show_utterances": False,  # 输出分句信息和时间戳，非常重要
                "enable_speaker_info": enable_speaker_info  # 说话人分离
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.submit_url, headers=submit_headers, json=submit_payload) as submit_resp:
                # 提交接口的 body 是空的，状态码在 Header 里
                status_code = submit_resp.headers.get("X-Api-Status-Code")
                message = submit_resp.headers.get("X-Api-Message")

                if status_code != "20000000":
                    raise Exception(f"ASR 提交任务失败! 状态码: {status_code}, 信息: {message}")

            # ---------------- 2. 轮询查询阶段 ----------------
            query_headers = {
                "X-Api-App-Key": self.app_key,
                "X-Api-Access-Key": self.access_key,
                "X-Api-Resource-Id": self.resource_id,
                "X-Api-Request-Id": task_id,
                "Content-Type": "application/json"
            }

            start_time = time.time()

            while time.time() - start_time < timeout:
                await asyncio.sleep(poll_interval)

                async with session.post(self.query_url, headers=query_headers, json={}) as query_resp:
                    q_status_code = query_resp.headers.get("X-Api-Status-Code")

                    if q_status_code == "20000000":
                        # 处理成功，解析返回的 JSON (包含 result 和 utterances)
                        data = await query_resp.json()
                        return data.get("result")

                    elif q_status_code in ["20000001", "20000002"]:
                        # 20000001: 处理中 / 20000002: 队列中
                        continue

                    elif q_status_code == "20000003":
                        # 静音音频，没有检测到人声
                        return {"text": "", "utterances": []}

                    else:
                        q_message = query_resp.headers.get("X-Api-Message")
                        raise Exception(f"ASR 查询任务报错! 状态码: {q_status_code}, 信息: {q_message}")

            raise TimeoutError(f"ASR 任务处理超时（超过 {timeout} 秒）")

class DoubaoPipeline():
    def __init__(self):
        self.client = ASRClient(app_key="5407910158", access_key="nITwxHAKsFx6bh5cA-OYbCdxF06EhdwC")

    def chat_template(self, text, audio):
        return {
            'audio': audio,
            'text': text
        }

    def run_batch(self, conversations, max_workers=1):
        async def _run_async():
            sem = asyncio.Semaphore(max_workers)

            async def process_single(c):
                async with sem:
                    try:
                        result = await self.client.asr(
                            audio_path=c['audio'],
                            audio_format="wav",
                            context=c['text'],
                        )
                        return result["text"]
                    except Exception as e:
                        print(e)
                        return None

            tasks = [process_single(c) for c in conversations]
            return await asyncio.gather(*tasks)

        return asyncio.run(_run_async())