#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估生成的决策训练数据集质量。
统计问题类型分布、gold/teacher 答案一致性等指标。
"""

import json
import sys
from collections import Counter
from typing import List, Dict


def _parse_field(s: Dict, field: str):
    """解析样本中 questions/gold/teacher 字段（新格式为 JSON 字符串）。

    字段缺失或不是合法 JSON 时返回 None。
    """
    v = s.get(field)
    if not isinstance(v, str):
        return None
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return None


def evaluate_json(filepath: str):
    """评估 JSON 文件的质量。返回 (result, samples)。"""
    with open(filepath, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not samples:
        return {"error": "文件中没有有效数据"}, []

    # 问题类型分布
    q_types = Counter()
    for s in samples:
        questions = _parse_field(s, "questions")
        if questions is None:
            continue
        for q_key, q_def in questions.items():
            q_types[q_def["type"]] += 1

    # Gold 答案分布
    gold_choices = Counter()
    teacher_choices = Counter()
    for s in samples:
        gold = _parse_field(s, "gold") or {}
        teacher = _parse_field(s, "teacher") or {}
        for q_key, gold_val in gold.items():
            gold_choices[gold_val] += 1
        for q_key, t_val in teacher.items():
            if not isinstance(t_val, dict):
                continue  # teacher 降级为 null
            if "noul" in t_val:
                teacher_choices[t_val["noul"]] += 1
            elif "probabilities" in t_val:
                max_key = max(t_val["probabilities"], key=t_val["probabilities"].get)
                teacher_choices[max_key] += 1

    # Teacher 与 Gold 一致性
    agreement = 0
    total = 0
    for s in samples:
        gold = _parse_field(s, "gold") or {}
        teacher = _parse_field(s, "teacher") or {}
        for q_key in gold:
            if q_key in teacher:
                gold_val = gold[q_key]
                t_val = teacher[q_key]
                if not isinstance(t_val, dict):
                    continue  # teacher 降级为 null，不计入一致率
                if "noul" in t_val:
                    pred = True if t_val["noul"] > 0.5 else False
                    if pred == gold_val:
                        agreement += 1
                    total += 1
                elif "probabilities" in t_val:
                    max_key = max(t_val["probabilities"], key=t_val["probabilities"].get)
                    if max_key == gold_val:
                        agreement += 1
                    total += 1

    result = {
        "total_samples": len(samples),
        "question_type_distribution": dict(q_types),
        "gold_answer_distribution": dict(gold_choices),
        "teacher_prediction_distribution": dict(teacher_choices),
        "agreement_rate": agreement / total if total > 0 else 0,
    }

    return result, samples


def validate_schema(samples):
    """校验样本是否符合参考文件结构。返回错误列表（空 = 通过）。"""
    errors = []
    for i, s in enumerate(samples):
        parsed = {}
        for field in ("questions", "gold", "teacher"):
            v = s.get(field)
            if not isinstance(v, str):
                errors.append(f"#{i}: {field} 应为 JSON 字符串")
                continue
            try:
                parsed[field] = json.loads(v)
            except json.JSONDecodeError:
                errors.append(f"#{i}: {field} 不是合法 JSON")
        if "questions" not in parsed or "gold" not in parsed:
            continue
        questions, gold = parsed["questions"], parsed["gold"]
        for k, q in questions.items():
            if q.get("type") not in ("choice", "noul", "score"):
                errors.append(f"#{i}: 问题 {k} 类型非法: {q.get('type')}")
                continue
            g = gold.get(k)
            if q["type"] == "noul" and not isinstance(g, bool):
                errors.append(f"#{i}: gold[{k}] noul 应为 bool，实际 {g!r}")
            elif q["type"] == "score" and (not isinstance(g, int) or isinstance(g, bool)):
                errors.append(f"#{i}: gold[{k}] score 应为整数索引，实际 {g!r}")
            elif q["type"] == "choice" and not isinstance(g, str):
                errors.append(f"#{i}: gold[{k}] choice 应为字符串键，实际 {g!r}")
    return errors


def print_evaluation(result: Dict, samples=None):
    """打印评估结果。"""
    if "error" in result:
        print(f"[错误] {result['error']}")
        return

    print("=" * 60)
    print("数据集质量评估报告")
    print("=" * 60)

    errors = validate_schema(samples or [])
    print(f"\nSchema 校验: {'通过' if not errors else f'{len(errors)} 个错误'}")
    for e in errors[:10]:
        print(f"  {e}")

    print(f"\n总样本数: {result['total_samples']}")

    print(f"\n问题类型分布:")
    for qtype, count in sorted(result["question_type_distribution"].items()):
        pct = count / sum(result["question_type_distribution"].values()) * 100
        print(f"  {qtype}: {count} ({pct:.1f}%)")

    print(f"\nGold 答案分布:")
    for ans, count in sorted(result["gold_answer_distribution"].items(), key=str):
        pct = count / sum(result["gold_answer_distribution"].values()) * 100
        print(f"  {ans}: {count} ({pct:.1f}%)")

    print(f"\nTeacher 预测分布:")
    for ans, count in sorted(result["teacher_prediction_distribution"].items(), key=str):
        pct = count / sum(result["teacher_prediction_distribution"].values()) * 100
        print(f"  {ans}: {count} ({pct:.1f}%)")

    print(f"\nTeacher-Gold 一致率: {result['agreement_rate']:.2%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="评估决策训练数据集质量")
    parser.add_argument("--input", default="dataset_10.json", help="JSON 文件路径")
    args = parser.parse_args()

    result, samples = evaluate_json(args.input)
    print_evaluation(result, samples)
