"""Teacher 概率预测的校验、归一化与 ensemble 均值。"""


def _match_score_index(criteria, key):
    """把名称型键匹配到 score criteria 索引（按文本相等）。"""
    for i, item in enumerate(criteria):
        if str(item).lower() == str(key).lower():
            return i
    return None


def validate_teacher(data, template):
    """校验并归一化一次 teacher 原始响应；非法返回 None。

    - noul: {"noul": float in [0,1]}
    - choice: probabilities 键必须恰好等于 criteria 键
    - score: probabilities 键必须恰好是 "0".."n-1"（名称键自动映射为索引）
    概率总和归一化为 1.0。
    """
    if not isinstance(data, dict):
        return None
    qtype = template["type"]
    if qtype == "noul":
        v = data.get("noul")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
            return {"noul": float(v)}
        return None

    probs = data.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return None
    criteria = template["criteria"]
    norm = {}
    for k, v in probs.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        if qtype == "score":
            keys = {str(i) for i in range(len(criteria))}
            if str(k) in keys:
                norm[str(k)] = float(v)
            else:
                idx = _match_score_index(criteria, str(k))
                if idx is None:
                    return None
                norm[str(idx)] = float(v)
        else:  # choice
            if k in criteria:
                norm[k] = float(v)
            else:
                return None

    expected = ({str(i) for i in range(len(criteria))} if qtype == "score"
                else set(criteria.keys()))
    if set(norm.keys()) != expected:
        return None
    total = sum(norm.values())
    if total <= 0:
        return None
    return {"probabilities": {k: v / total for k, v in norm.items()}}


def average_teacher(samples, template):
    """对多次校验通过的 teacher 响应取均值（ensemble）。空输入返回 None。"""
    samples = [s for s in samples if s]
    if not samples:
        return None
    if template["type"] == "noul":
        return {"noul": sum(s["noul"] for s in samples) / len(samples)}
    acc = {}
    for s in samples:
        for k, v in s["probabilities"].items():
            acc[k] = acc.get(k, 0.0) + v
    return {"probabilities": {k: v / len(samples) for k, v in acc.items()}}
