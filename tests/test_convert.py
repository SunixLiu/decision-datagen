import json
import os

from convert_for_finetune import load_samples, build_record


SAMPLE = {
    "state_id": "synth-x", "generator": "v1-hr", "domain": "hr", "n_questions": 1,
    "state": "{\"flag\": true}", "state_is_json": True,
    "questions": json.dumps({"q1": {"type": "noul", "instructions": "x",
                                    "criteria": {"true": "", "false": ""}}}, ensure_ascii=False),
    "gold": json.dumps({"q1": True}, ensure_ascii=False),
    "teacher": json.dumps({"q1": {"noul": 0.9}}, ensure_ascii=False),
    "teacher_model": "m",
}


def test_load_samples_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(SAMPLE, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_samples(str(p)) == [SAMPLE]


def test_load_samples_json_array(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([SAMPLE], ensure_ascii=False), encoding="utf-8")
    assert load_samples(str(p)) == [SAMPLE]


def test_build_record():
    rec = build_record(SAMPLE)
    assert rec["messages"][0]["role"] == "user"
    assert rec["messages"][1]["role"] == "assistant"
    assert "q1" in rec["messages"][0]["content"]
    assert json.loads(rec["messages"][1]["content"].split("【决策结果】\n", 1)[1]) == {"q1": True}
