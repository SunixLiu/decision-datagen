from gold_rules import derive_gold, get_field


def test_get_field_nested():
    assert get_field({"a": {"b": 1}}, "a.b") == 1
    assert get_field({"a": 1}, "a.b") is None
    assert get_field({}, "x") is None


def test_count_gte_true():
    t = {"type": "noul", "criteria": {"true": "", "false": ""},
         "facts": {"field": "risk_indicators.aml_flags", "rule": {"op": "count_gte", "value": 2}}}
    assert derive_gold({"risk_indicators": {"aml_flags": ["a", "b"]}}, t) is True


def test_count_gte_false():
    t = {"type": "noul", "criteria": {"true": "", "false": ""},
         "facts": {"field": "flags", "rule": {"op": "count_gte", "value": 2}}}
    assert derive_gold({"flags": ["a"]}, t) is False


def test_map_score_index():
    t = {"type": "score", "criteria": ["a", "b", "c", "d"],
         "facts": {"field": "x.level", "rule": {"op": "map", "mapping": {"full": 3}}}}
    assert derive_gold({"x": {"level": "full"}}, t) == 3


def test_map_choice_key():
    t = {"type": "choice", "criteria": {"compliant": "", "non-compliant": ""},
         "facts": {"field": "status", "rule": {"op": "map", "mapping": {"ok": "compliant"}}}}
    assert derive_gold({"status": "ok"}, t) == "compliant"


def test_identity_choice():
    t = {"type": "choice", "criteria": {"compliant": "", "non-compliant": ""},
         "facts": {"field": "status", "rule": {"op": "identity"}}}
    assert derive_gold({"status": "compliant"}, t) == "compliant"
    assert derive_gold({"status": "bogus"}, t) is None  # 不在 criteria 内 → None


def test_gte_numeric():
    t = {"type": "noul", "criteria": {"true": "", "false": ""},
         "facts": {"field": "similarity", "rule": {"op": "gte", "value": 0.2}}}
    assert derive_gold({"similarity": 0.25}, t) is True
    assert derive_gold({"similarity": 0.19}, t) is False


def test_in_list():
    t = {"type": "noul", "criteria": {"true": "", "false": ""},
         "facts": {"field": "flags", "rule": {"op": "in", "value": "shell"}}}
    assert derive_gold({"flags": ["shell", "x"]}, t) is True
    assert derive_gold({"flags": ["x"]}, t) is False


def test_missing_field_returns_none():
    t = {"type": "noul", "criteria": {"true": "", "false": ""},
         "facts": {"field": "nope", "rule": {"op": "eq", "value": True}}}
    assert derive_gold({}, t) is None


def test_no_facts_returns_none():
    assert derive_gold({}, {"type": "noul", "criteria": {}}) is None
