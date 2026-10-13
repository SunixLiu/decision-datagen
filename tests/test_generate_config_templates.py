from generate_config import filter_valid_templates


class StubLLM:
    def chat_json(self, messages, temperature=0.7, max_tokens=4096):
        return {
            "decision_points": ["p1"],
            "templates": [
                {"key": "new_q", "type": "noul", "instructions": "x",
                 "criteria": {"true": "", "false": ""},
                 "facts": {"field": "f", "rule": {"op": "eq", "value": True}},
                 "policy_keywords": ["K"]},
                {"key": "dup_key", "type": "noul", "instructions": "x",
                 "criteria": {"true": "", "false": ""},
                 "facts": {"field": "f", "rule": {"op": "eq", "value": True}}},
                {"key": "bad_type", "type": "weird", "instructions": "x",
                 "criteria": {}, "facts": {"field": "f", "rule": {}}},
                {"key": "no_facts", "type": "noul", "instructions": "x",
                 "criteria": {"true": "", "false": ""}},
            ],
        }


def test_filter_valid_templates():
    existing_keys = {"dup_key"}
    valid = filter_valid_templates(StubLLM().chat_json([{"role": "user", "content": "p"}]),
                                   existing_keys)
    assert [t["key"] for t in valid] == ["new_q"]
