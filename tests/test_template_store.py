import json
import os

import pytest

from template_store import load_templates, merge_templates, save_templates


def test_merge_dedup_by_key():
    existing = [{"key": "a", "type": "noul"}]
    new = [{"key": "a", "type": "choice"}, {"key": "b", "type": "score"}]
    merged, added = merge_templates(existing, new)
    assert [t["key"] for t in merged] == ["a", "b"]
    assert added == 1


def test_save_and_load_roundtrip(tmp_path):
    templates = [{"key": "x", "type": "noul", "criteria": {"true": "", "false": ""}}]
    save_templates(templates, "hr", base=str(tmp_path))
    loaded = load_templates("HR", base=str(tmp_path))  # 大小写不敏感
    assert loaded == templates


def test_load_falls_back_to_default(tmp_path):
    (tmp_path / "templates").mkdir()
    with open(tmp_path / "templates" / "default.json", "w", encoding="utf-8") as f:
        json.dump([{"key": "d"}], f)
    assert load_templates("unknown-domain", base=str(tmp_path)) == [{"key": "d"}]


def test_load_missing_raises(tmp_path):
    (tmp_path / "templates").mkdir()
    with pytest.raises(FileNotFoundError):
        load_templates("nope", base=str(tmp_path))
