"""OpenAI 兼容 LLM API 客户端（从 generate_dataset.py 抽出，供两个脚本共享）。"""
import json
import os
import re
import sys
from typing import List
import requests


def _ensure_utf8_stdout():
    """Windows 控制台默认 GBK；重配置为 UTF-8，避免打印含特殊字符的文本时崩溃。"""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_ensure_utf8_stdout()


class LLMClient:
    """轻量级 LLM API 客户端，支持 OpenAI 兼容接口。"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = "Qwen3.6-35B-A3B"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
        self.model = model or os.getenv("MODEL_NAME", "Qwen3.6-35B-A3B")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        self.last_finish_reason = None

    def chat(self, messages: List[dict], temperature=0.7, max_tokens=4096):
        """发送聊天请求并返回响应文本。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = f"{self.base_url}/chat/completions"
        resp = self.session.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("choices"):
            self.last_finish_reason = None
            return ""
        choice = data["choices"][0]
        self.last_finish_reason = choice.get("finish_reason")
        content = choice.get("message", {}).get("content", "") or ""
        # 清理 markdown 代码块标记
        if content and content.strip().startswith("```"):
            lines = content.split("\n")
            cleaned = []
            skip = False
            for line in lines:
                if line.strip().startswith("```"):
                    skip = not skip
                    continue
                if not skip:
                    cleaned.append(line)
            content = "\n".join(cleaned).strip()
        return content

    def chat_json(self, messages: List[dict], temperature=0.7, max_tokens=4096, retries: int = 3):
        """发送聊天请求并返回解析后的 JSON 对象，支持重试。"""
        prompt_text = ""
        for m in messages:
            prompt_text += str(m.get("content", ""))
        print(f" [DEBUG] Prompt length: {len(prompt_text)} chars", flush=True)
        print(f" [DEBUG] Prompt preview: {prompt_text[:200]}...", flush=True)
        current_max = max_tokens
        for attempt in range(retries):
            try:
                content = self.chat(messages, temperature=temperature, max_tokens=current_max)
                finish = self.last_finish_reason
                if finish == "length":
                    # 输出被截断：原样重试必然再截断，改为升档 max_tokens
                    if current_max < 32768:
                        current_max = min(current_max * 2, 32768)
                        print(f" [DEBUG] Attempt {attempt+1} 被截断 (finish_reason=length)，max_tokens 升至 {current_max}", flush=True)
                        continue
                    raise ValueError("输出在 max_tokens 上限仍被截断")
                print(f" [DEBUG] Attempt {attempt+1}: raw response length={len(content) if content else 0}", flush=True)
                if content:
                    print(f" [DEBUG] Response preview: {content[:200]}...", flush=True)
                if not content or not content.strip():
                    if attempt < retries - 1:
                        print(f" [DEBUG] Attempt {attempt+1} returned empty, retrying...", flush=True)
                        continue
                    raise ValueError("LLM 返回空内容")

                # 尝试直接解析
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass

                # 尝试从文本中提取 JSON（使用括号匹配）
                start = content.find("{")
                if start >= 0:
                    depth = 0
                    end = start
                    in_string = False
                    escape = False
                    for i in range(start, len(content)):
                        c = content[i]
                        if in_string:
                            if escape:
                                escape = False
                            elif c == "\\":
                                escape = True
                            elif c == '"':
                                in_string = False
                        else:
                            if c == '"':
                                in_string = True
                            elif c == "{":
                                depth += 1
                            elif c == "}":
                                depth -= 1
                                if depth == 0:
                                    end = i + 1
                                    break
                    json_str = content[start:end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

                # 尝试修复不完整的 JSON
                match = re.search(r'\{.*', content, re.DOTALL)
                if match:
                    json_str = match.group()
                    open_count = json_str.count("{")
                    close_count = json_str.count("}")
                    if open_count > close_count:
                        json_str += "}" * (open_count - close_count)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

                if attempt < retries - 1:
                    continue
                raise ValueError(f"无法解析 LLM 返回的 JSON: {content[:500]}")
            except Exception as e:
                if attempt < retries - 1:
                    continue
                raise ValueError(f"无法解析 LLM 返回的 JSON: {str(e)[:200]}")
