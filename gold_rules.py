"""从 state 内置事实确定性推导 gold 答案。

rule 语法（facts.rule.op）：
- identity: gold = state[field]（choice；值必须是 criteria 的键）
- map:      gold = mapping[state[field]]（score → 整数索引；choice → 键）
- count_gte: true 当 len(state[field]) >= value（noul）
- gte / lte: 数值比较（noul）
- eq:       state[field] == value（noul）
- in:       value in state[field]（noul）
"""


def get_field(state, path):
    """按点分路径读取嵌套字段；缺失返回 None。"""
    cur = state
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def derive_gold(state, template):
    """推导 gold 答案；无法推导时返回 None。

    返回值类型：noul → bool；choice → criteria 键(str)；score → int 索引。
    """
    facts = template.get("facts")
    if not facts or "field" not in facts or "rule" not in facts:
        return None
    value = get_field(state, facts["field"])
    if value is None:
        return None
    rule = facts["rule"]
    op = rule.get("op")
    criteria = template["criteria"]
    qtype = template["type"]

    if op == "identity":
        if isinstance(criteria, dict) and value in criteria:
            return value
        return None

    if op == "map":
        mapping = rule.get("mapping", {})
        if value in mapping:
            ans = mapping[value]
            if qtype == "score" and isinstance(ans, int) and not isinstance(ans, bool) \
                    and 0 <= ans < len(criteria):
                return ans
            if isinstance(criteria, dict) and ans in criteria:
                return ans
        return None

    if op == "count_gte":
        n = len(value) if isinstance(value, (list, tuple)) else (1 if value else 0)
        return bool(n >= rule["value"])

    if op == "gte":
        return bool(value >= rule["value"])

    if op == "lte":
        return bool(value <= rule["value"])

    if op == "eq":
        return bool(value == rule["value"])

    if op == "in":
        return bool(rule["value"] in value)

    return None
