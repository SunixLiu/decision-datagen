import json

from generate_dataset import process_sample


class StubLLM:
    """返回批量答案：解析 prompt 中的问题ID，逐题给 noul 概率。"""
    model = "stub-model"

    def __init__(self):
        self.n_calls = 0

    def chat_json(self, messages, temperature=0.7, max_tokens=4096):
        self.n_calls += 1
        prompt = messages[-1]["content"]
        keys = [line[len("问题ID: "):] for line in prompt.splitlines()
                if line.startswith("问题ID: ")]
        return {"answers": {k: {"noul": 0.9} for k in keys}}


NOUL_T = {
    "key": "q1", "type": "noul", "instructions": "x",
    "criteria": {"true": "", "false": ""},
    "facts": {"field": "flag", "rule": {"op": "eq", "value": True}},
}

NOUL_T2 = {
    "key": "q2", "type": "noul", "instructions": "y",
    "criteria": {"true": "", "false": ""},
    "facts": {"field": "flag2", "rule": {"op": "eq", "value": True}},
}


def test_process_sample_gold_and_reference_format():
    llm = StubLLM()
    s = process_sample({"flag": True}, [NOUL_T], "hr", llm, {"q1": "digest"}, 1)
    assert isinstance(s["questions"], str)
    assert isinstance(s["gold"], str)
    assert isinstance(s["teacher"], str)
    assert json.loads(s["gold"]) == {"q1": True}
    assert json.loads(s["teacher"]) == {"q1": {"noul": 0.9}}
    assert s["generator"] == "v1-hr"
    assert s["state_is_json"] is True
    assert s["n_questions"] == 1


def test_process_sample_batches_questions_in_one_call():
    llm = StubLLM()
    s = process_sample({"flag": True, "flag2": True}, [NOUL_T, NOUL_T2], "hr", llm, {}, 1)
    assert llm.n_calls == 1  # 批量：一次调用回答两个问题
    assert json.loads(s["teacher"]) == {"q1": {"noul": 0.9}, "q2": {"noul": 0.9}}


def test_process_sample_drops_missing_fact():
    assert process_sample({}, [NOUL_T], "hr", StubLLM(), {}, 1) is None


def test_process_sample_ensemble_label():
    class MultiLLM(StubLLM):
        model = "big-model"
    s = process_sample({"flag": True}, [NOUL_T], "hr", MultiLLM(), {"q1": "d"}, 3)
    assert s["teacher_model"] == "big-model x3 mean"
