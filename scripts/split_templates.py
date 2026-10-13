"""一次性迁移：把 question_templates.json 拆分为 templates/{domain}.json。"""
import json
import os

SRC = "question_templates.json"
OUT_DIR = "templates"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)
    for domain, templates in data.items():
        if domain == "_comment":
            continue
        path = os.path.join(OUT_DIR, f"{domain}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)
        print(f"wrote {path} ({len(templates)} templates)")


if __name__ == "__main__":
    main()
