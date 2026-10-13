from teacher import average_teacher, validate_teacher


NOUL = {"type": "noul", "criteria": {"true": "", "false": ""}}
CHOICE = {"type": "choice", "criteria": {"a": "", "b": ""}}
SCORE = {"type": "score", "criteria": ["低风险", "中等风险", "高风险"]}


def test_noul_valid():
    assert validate_teacher({"noul": 0.8}, NOUL) == {"noul": 0.8}


def test_noul_invalid():
    assert validate_teacher({"noul": "high"}, NOUL) is None
    assert validate_teacher({"noul": 1.5}, NOUL) is None
    assert validate_teacher({}, NOUL) is None


def test_choice_exact():
    r = validate_teacher({"probabilities": {"a": 0.7, "b": 0.3}}, CHOICE)
    assert r == {"probabilities": {"a": 0.7, "b": 0.3}}


def test_choice_normalizes_sum():
    r = validate_teacher({"probabilities": {"a": 0.7, "b": 0.7}}, CHOICE)
    assert abs(r["probabilities"]["a"] - 0.5) < 1e-9
    assert abs(r["probabilities"]["b"] - 0.5) < 1e-9


def test_choice_missing_key_rejected():
    assert validate_teacher({"probabilities": {"a": 1.0}}, CHOICE) is None
    assert validate_teacher({"probabilities": {"a": 1.0, "c": 0.0}}, CHOICE) is None


def test_score_index_keys():
    r = validate_teacher({"probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}}, SCORE)
    assert set(r["probabilities"]) == {"0", "1", "2"}


def test_score_name_keys_mapped_to_indices():
    r = validate_teacher({"probabilities": {"低风险": 0.5, "中等风险": 0.3, "高风险": 0.2}}, SCORE)
    assert r["probabilities"] == {"0": 0.5, "1": 0.3, "2": 0.2}


def test_average_ensemble_choice():
    r = average_teacher(
        [{"probabilities": {"a": 0.9, "b": 0.1}},
         {"probabilities": {"a": 0.7, "b": 0.3}}], CHOICE)
    assert abs(r["probabilities"]["a"] - 0.8) < 1e-9


def test_average_ensemble_noul():
    r = average_teacher([{"noul": 0.9}, {"noul": 0.5}], NOUL)
    assert abs(r["noul"] - 0.7) < 1e-9


def test_average_empty_returns_none():
    assert average_teacher([], CHOICE) is None
