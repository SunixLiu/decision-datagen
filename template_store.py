"""按领域保存问题模板：templates/{domain}.json，兜底 templates/default.json。"""
import json
import os


def load_templates(domain, base="."):
    """加载领域模板列表；不存在则回退 default.json。"""
    d = os.path.join(base, "templates")
    path = os.path.join(d, f"{domain.lower()}.json")
    if not os.path.exists(path):
        path = os.path.join(d, "default.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"领域 '{domain}' 无模板文件且 {d}/default.json 不存在")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_templates(templates, domain, base="."):
    """保存模板列表到 templates/{domain}.json，返回路径。"""
    d = os.path.join(base, "templates")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{domain.lower()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)
    return path


def merge_templates(existing, new):
    """把新模板追加进现有列表，按 key 去重。返回 (merged_list, added_count)。"""
    by_key = {t["key"]: t for t in existing if t.get("key")}
    added = 0
    for t in new:
        key = t.get("key")
        if key and key not in by_key:
            by_key[key] = t
            added += 1
    return list(by_key.values()), added
