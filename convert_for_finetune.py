#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将生成的 JSON 训练数据转换为 OpenAI fine-tuning 格式。
每条样本转换为 {messages: [{role: user, content: ...}, {role: assistant, content: ...}]}
"""

import json
import sys
from typing import List, Dict


def load_samples(input_file):
    """加载 JSON 数组或 JSONL 格式的样本。"""
    if input_file.endswith(".jsonl"):
        with open(input_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(input_file, encoding="utf-8") as f:
        return json.load(f)


def build_record(sample):
    """把一条样本转换为 OpenAI fine-tuning messages 记录。"""
    questions = json.loads(sample["questions"])
    gold = json.loads(sample["gold"])
    state = sample["state"]
    user_content = (
        "请根据政策文档对以下申请/事件状态进行决策判断。\n\n"
        "【状态】\n" + state + "\n\n"
        "【问题】\n"
    )
    for q_key, q_def in questions.items():
        user_content += (f"\n问题: {q_key}\n说明: {q_def['instructions']}\n"
                        f"选项: {json.dumps(q_def['criteria'], ensure_ascii=False)}\n")
    assistant_content = "【决策结果】\n" + json.dumps(gold, ensure_ascii=False)
    return {"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]}


def convert_json_to_openai_format(input_file: str, output_file: str):
    """将 JSON 数组或 JSONL 文件转换为 OpenAI fine-tuning JSONL 格式。"""
    samples = load_samples(input_file)

    count = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for sample in samples:
            record = build_record(sample)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"转换完成：{count} 条样本已保存至 {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="将 JSON 训练数据转换为 OpenAI fine-tuning 格式")
    parser.add_argument("--input", default="dataset_10.json", help="输入的 JSON 文件路径")
    parser.add_argument("--output", default="finetune_data.jsonl", help="输出的 JSONL 文件路径")
    args = parser.parse_args()

    convert_json_to_openai_format(args.input, args.output)
