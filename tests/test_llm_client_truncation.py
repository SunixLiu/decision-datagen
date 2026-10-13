from llm_client import LLMClient


class TruncatingClient(LLMClient):
    """模拟服务端：max_tokens 不足时返回截断（finish_reason=length）。"""

    def __init__(self):
        super().__init__(api_key="x", base_url="http://localhost/v1")
        self.calls = []

    def chat(self, messages, temperature=0.7, max_tokens=4096):
        self.calls.append(max_tokens)
        if max_tokens < 8192:
            self.last_finish_reason = "length"
            return '{"partial": '
        self.last_finish_reason = "stop"
        return '{"answers": {"q1": {"noul": 0.5}}}'


def test_chat_json_escalates_max_tokens_on_length():
    c = TruncatingClient()
    out = c.chat_json([{"role": "user", "content": "p"}], max_tokens=4096)
    assert out == {"answers": {"q1": {"noul": 0.5}}}
    assert c.calls == [4096, 8192]  # 第一次截断 → 升档后成功


def test_chat_json_no_escalation_when_not_truncated():
    class OkClient(TruncatingClient):
        def chat(self, messages, temperature=0.7, max_tokens=4096):
            self.calls.append(max_tokens)
            self.last_finish_reason = "stop"
            return '{"ok": true}'

    c = OkClient()
    out = c.chat_json([{"role": "user", "content": "p"}], max_tokens=4096)
    assert out == {"ok": True}
    assert c.calls == [4096]  # 未截断 → 不升档、一次成功
