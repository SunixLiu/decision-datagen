from evaluate_dataset import validate_schema


def make_sample(gold_obj, questions_obj):
    import json
    return {
        "state_id": "synth-x", "generator": "v1-hr", "domain": "hr",
        "n_questions": len(questions_obj),
        "state": "{}", "state_is_json": True,
        "questions": json.dumps(questions_obj, ensure_ascii=False),
        "gold": json.dumps(gold_obj, ensure_ascii=False),
        "teacher": json.dumps({}, ensure_ascii=False),
        "teacher_model": "m",
    }


Q = {"q1": {"type": "noul", "instructions": "", "criteria": {"true": "", "false": ""}},
     "q2": {"type": "score", "instructions": "", "criteria": ["a", "b"]},
     "q3": {"type": "choice", "instructions": "", "criteria": {"x": "", "y": ""}}}


def test_valid_sample_no_errors():
    s = make_sample({"q1": True, "q2": 1, "q3": "x"}, Q)
    assert validate_schema([s]) == []


def test_gold_type_mismatch_detected():
    s = make_sample({"q1": "yes", "q2": 1, "q3": "x"}, Q)   # noul gold 应为 bool
    errs = validate_schema([s])
    assert any("q1" in e for e in errs)


def test_score_gold_must_be_int():
    s = make_sample({"q1": True, "q2": "a", "q3": "x"}, Q)
    errs = validate_schema([s])
    assert any("q2" in e for e in errs)


def test_questions_not_string_detected():
    s = make_sample({"q1": True, "q2": 1, "q3": "x"}, Q)
    s["questions"] = Q  # 应为字符串
    errs = validate_schema([s])
    assert any("questions" in e for e in errs)
