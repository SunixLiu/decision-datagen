#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策型模型训练数据集生成程序
基于 @docsrc/Compliance-Manual-2018.pdf 合规手册生成结构化决策训练数据

目标格式参考：sql-console-for-n4ze3m-typed-decisions-synth.json
每条样本包含：state_id, generator, domain, n_questions, state, state_is_json,
questions (type/criteria), gold, teacher

Jev 三原语类型：choice（多选一）、noul（是非判断）、score（评分）
"""

import json
import os
import re
import sys
import uuid
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import requests

from llm_client import LLMClient
from config_loader import load_model_config
from gold_rules import derive_gold
from template_store import load_templates
from teacher import average_teacher, validate_teacher


# ---------------------------------------------------------------------------
# 1. 从 PDF 提取文本
# ---------------------------------------------------------------------------
def extract_policy_text(pdf_path: str = "docsrc/Compliance-Manual-2018.pdf",
                        txt_path: str = "docsrc/Compliance-Manual-2018.txt") -> str:
    """提取 PDF 中的文本并保存为 .txt；若已存在则直接读取。"""
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[PDF 文本提取] 已保存 {len(text)} 字符到 {txt_path}")
        return text
    except ImportError:
        raise ImportError("pdfminer.six is required. Install with: pip install pdfminer.six")


# ---------------------------------------------------------------------------
# 2. 提取合规手册中的关键章节
# ---------------------------------------------------------------------------
RELEVANT_SECTION_KEYWORDS = [
    "Fiduciary Duty", "Duty to Supervise", "Compliance Risk Assessment",
    "Registration and Licensing", "Form ADV Disclosure", "Client Contracts",
    "Books and Records", "Diminished Capacity", "Custody", "Anti Money Laundering",
    "Proxy Voting", "Advertising", "Social Networking", "Outside Employment",
    "Correspondence", "Email and Other Electronic Communications", "Complaints",
    "Portfolio Management", "Suitability", "Account Opening", "Account Closing",
    "Federal Filing Requirements", "Best Execution", "Trading", "Trade Error",
    "Economic Benefits", "ERISA", "Privacy Policy", "Information Security",
    "Cybersecurity", "Fraudulent Transfers", "Identity Theft", "Business Continuity",
]


def split_sections(policy_text):
    """按标题行切分文档，返回 [(title, content), ...]。"""
    lines = policy_text.split("\n")
    sections = []
    current_title, current_lines = None, []
    for line in lines:
        stripped = line.strip()
        is_heading = (
            stripped and len(stripped) < 100
            and stripped[0].isupper()
            and not stripped.startswith("•")
            and not stripped.startswith("-")
            and "\t" not in stripped
            and not stripped.endswith(".")
        )
        if is_heading:
            if current_title and current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title, current_lines = stripped, []
        else:
            if current_title:
                current_lines.append(line)
    if current_title and current_lines:
        sections.append((current_title, "\n".join(current_lines)))
    return sections


def extract_relevant_sections(policy_text, max_chars=30000, keywords=None):
    """从政策文档中提取与决策判断最相关的章节（保持原行为）。"""
    if keywords is None:
        keywords = RELEVANT_SECTION_KEYWORDS
    sections = split_sections(policy_text)
    relevant = [(t, c) for t, c in sections
                if any(kw.lower() in t.lower() for kw in keywords)]
    if not relevant:
        relevant = sections[:20]
    parts, total = [], 0
    for title, content in relevant:
        section_text = f"\n\n=== {title} ===\n{content}"
        if total + len(section_text) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                section_text = section_text[:remaining] + "\n... (truncated)"
            else:
                break
        parts.append(section_text)
        total += len(section_text)
    return "\n".join(parts)


def build_policy_digest(policy_text, templates, max_chars=3000):
    """为每个问题预切相关政策章节片段（替代每次嵌入全文）。

    templates: {key: template_dict}，template 可含 policy_keywords。
    无匹配关键词时回退文档开头 max_chars 字符。
    """
    sections = split_sections(policy_text)
    digest = {}
    for key, t in templates.items():
        kws = t.get("policy_keywords") or []
        matched = [f"=== {title} ===\n{content}"
                   for title, content in sections
                   if any(kw.lower() in title.lower() for kw in kws)]
        text = "\n\n".join(matched) if matched else policy_text
        digest[key] = text[:max_chars]
    return digest


# ---------------------------------------------------------------------------
# 3. 定义问题模板（基于合规手册政策）
# ---------------------------------------------------------------------------
def build_question_templates(domain: str = "compliance") -> Dict[str, dict]:
    """
    从 question_templates.json 加载指定领域的问题模板。
    每个问题包含：type (choice/noul/score), instructions, criteria

    支持的领域：compliance, healthcare, education, e-commerce, hr
    """
    import json as _json
    import os as _os

    # 查找模板文件（相对于脚本目录）
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    template_path = _os.path.join(script_dir, "question_templates.json")

    if not _os.path.exists(template_path):
        raise FileNotFoundError(f"模板文件不存在: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        all_templates = _json.load(f)

    domain_key = domain.lower()
    if domain_key in all_templates:
        raw_templates = all_templates[domain_key]
    elif "default" in all_templates:
        raw_templates = all_templates["default"]
    else:
        raise ValueError(f"不支持的领域: {domain}。可用领域: {list(all_templates.keys())}")

    # 转换为以 key 为键的字典
    templates = {}
    for t in raw_templates:
        templates[t["key"]] = {
            "type": t["type"],
            "instructions": t["instructions"],
            "criteria": t["criteria"],
        }

    return templates


# ---------------------------------------------------------------------------
# 4. 生成场景状态（基于合规手册）
# ---------------------------------------------------------------------------
def build_state_prompt(domain_config, pool_templates, batch_size, policy_text):
    """构造 state 生成提示词，并注入 ground-truth 字段要求。"""
    template = domain_config.get("state_prompt_template", "")
    prompt = template.replace("{batch_size}", str(batch_size))
    prompt = prompt.replace("{doc_name}", domain_config.get("doc_name", "政策文档"))
    prompt = prompt.replace("{policy_text}", policy_text)
    facts_lines = []
    for t in pool_templates:
        f = t["facts"]
        rule = f["rule"]
        rule_desc = rule.get("op") + (f" {rule['value']}" if "value" in rule else "")
        facts_lines.append(f"- 问题 {t['key']}：state 必须包含字段 `{f['field']}`（推导规则: {rule_desc}）")
    prompt += ("\n\n【Ground-truth 字段要求】\n"
              "每个 state 必须包含以下字段，其值将用于确定性推导答案：\n"
              + "\n".join(facts_lines))
    return prompt


def generate_states(llm, policy_text, domain, pool_templates, batch_size, max_tokens=None):
    """根据政策文档生成场景状态；pool_templates 决定注入的 ground-truth 字段。"""
    if max_tokens is None:
        max_tokens = 16384  # 模型会消耗 reasoning tokens，4096 不足以输出完整 states
    domain_config = load_domain_config(domain)
    prompt = build_state_prompt(domain_config, pool_templates, batch_size, policy_text)
    print(f" [DEBUG] Policy text length: {len(policy_text)} chars")
    print(f" [DEBUG] Final prompt length: {len(prompt)} chars")
    try:
        data = llm.chat_json(
            messages=[
                {"role": "system", "content": "你是一个专业的AI助手，请严格按照要求返回JSON格式的内容。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return data.get("states", []) if isinstance(data, dict) else []
    except Exception as e:
        print(f" [场景生成错误] JSON 解析失败: {e}")
        return []


# ---------------------------------------------------------------------------
# 5. 教师模型回答（probabilistic predictions）
# ---------------------------------------------------------------------------
def teacher_answer_questions(llm, digests, state, templates, ensemble_n=1, max_tokens=None):
    """批量教师预测：一次调用回答同一 state 的全部问题，逐 key 校验 + ensemble 均值。

    digests: {key: policy_digest_text}；templates: 待回答的模板列表。
    返回 {key: validated_dict | None}。
    """
    if max_tokens is None:
        max_tokens = 4096  # 模型消耗 reasoning tokens，过小会导致响应被截断为空
    state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    q_blocks = []
    for t in templates:
        q_blocks.append(
            f"问题ID: {t['key']}\n"
            f"类型: {t['type']}\n"
            f"说明: {t['instructions']}\n"
            f"选项/标准: {json.dumps(t['criteria'], ensure_ascii=False)}\n"
            f"【相关政策片段】\n{digests.get(t['key'], '')}"
        )
    prompt = (
        "你是一个高精度的决策模型教师引擎。请根据以下 Policy 文档片段，对给定的状态回答全部指定问题。\n\n"
        + "\n\n".join(q_blocks) + "\n\n"
        "【状态】\n" + state_str + "\n\n"
        "请严格按照以下 JSON Schema 输出：\n"
        '{"answers": {\n'
        '  "<问题ID>": {"noul": float}  // noul 类型\n'
        '  或 "<问题ID>": {"probabilities": {...}}  // choice/score 类型（score 用索引字符串 "0","1",... 为键）\n'
        "}\n}\n"
        "注意：每个问题的概率总和必须为 1.0。"
    )
    samples = []
    for _ in range(max(1, ensemble_n)):
        try:
            data = llm.chat_json(messages=[{"role": "user", "content": prompt}],
                                 temperature=0.3, max_tokens=max_tokens)
        except Exception as e:
            print(f" [教师回答错误] {e}")
            continue
        answers = data.get("answers", {}) if isinstance(data, dict) else {}
        sample = {}
        for t in templates:
            v = validate_teacher(answers.get(t["key"]), t)
            if v is not None:
                sample[t["key"]] = v
        if sample:
            samples.append(sample)
    result = {}
    for t in templates:
        vals = [s[t["key"]] for s in samples if t["key"] in s]
        result[t["key"]] = average_teacher(vals, t)
    return result


def process_sample(state, selected_templates, domain, teacher_llm, digest, ensemble_n, max_tokens=None):
    """处理单个 state：规则推导 gold + 批量 teacher 预测，构造参考格式样本。失败返回 None。"""
    questions, gold, derivable = {}, {}, []
    for t in selected_templates:
        g = derive_gold(state, t)
        if g is None:
            continue
        questions[t["key"]] = {"type": t["type"], "instructions": t["instructions"],
                               "criteria": t["criteria"]}
        gold[t["key"]] = g
        derivable.append(t)
    if not derivable:
        return None
    teacher = teacher_answer_questions(teacher_llm, digest, state, derivable,
                                       ensemble_n=ensemble_n, max_tokens=max_tokens)
    state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    model_label = teacher_llm.model + (f" x{ensemble_n} mean" if ensemble_n > 1 else "")
    return {
        "state_id": "synth-" + uuid.uuid4().hex[:16],
        "generator": f"v1-{domain}",
        "domain": domain,
        "n_questions": len(questions),
        "state": state_str,
        "state_is_json": not isinstance(state, str),
        "questions": json.dumps(questions, ensure_ascii=False),
        "gold": json.dumps(gold, ensure_ascii=False),
        "teacher": json.dumps(teacher, ensure_ascii=False),
        "teacher_model": model_label,
    }


# ---------------------------------------------------------------------------
# 6. 生成完整数据集
# ---------------------------------------------------------------------------
def generate_dataset(state_llm, teacher_llm, policy_text, domain, target_samples=50,
                     output_filename="dataset.json", batch_size=10, max_tokens=None,
                     questions_per_sample=3, concurrency=1, ensemble_n=1,
                     resume=False, templates_dir="."):
    """生成完整决策训练数据集（并发 + JSONL 增量写 + resume）。"""
    templates = load_templates(domain, base=templates_dir)
    fact_templates = [t for t in templates if t.get("facts")]
    if not fact_templates:
        raise SystemExit(f"领域 {domain} 没有带 facts 规格的模板；请先运行 generate_config.py")
    digest = build_policy_digest(policy_text, {t["key"]: t for t in fact_templates})
    jsonl_path = os.path.splitext(output_filename)[0] + ".jsonl"

    done = 0
    if resume and os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            done = sum(1 for line in f if line.strip())
        print(f"[resume] {jsonl_path} 已有 {done} 条样本")

    degraded = 0
    fail_streak = 0
    print(f"=== 开始生成 (目标 {target_samples}, concurrency={concurrency}, ensemble_n={ensemble_n}) ===")
    print(f"可用模板(带facts): {len(fact_templates)} / 总模板: {len(templates)}")

    with open(jsonl_path, "a", encoding="utf-8") as fout:
        while done < target_samples:
            batch = min(batch_size, target_samples - done)
            pool_size = min(2 * questions_per_sample, len(fact_templates))
            pool = random.sample(fact_templates, pool_size)
            states = generate_states(state_llm, policy_text, domain, pool, batch, max_tokens=max_tokens)
            if not states:
                fail_streak += 1
                print(f" [状态生成失败] 连续第 {fail_streak} 次")
                if fail_streak >= 3:
                    raise SystemExit("状态生成连续失败 3 次，中止（已完成部分保留在 JSONL）")
                continue
            fail_streak = 0
            tasks = []
            for state in states[:target_samples - done]:
                n_q = random.randint(max(1, questions_per_sample - 1),
                                    min(questions_per_sample, len(pool)))
                sel = random.sample(pool, n_q)
                tasks.append((state, sel))
            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
                futs = [ex.submit(process_sample, st, sel, domain, teacher_llm, digest,
                                  ensemble_n, max_tokens)
                        for st, sel in tasks]
                for fut in as_completed(futs):
                    sample = fut.result()
                    if sample is None:
                        degraded += 1
                        continue
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    fout.flush()
                    done += 1
                    print(f"  [成功生成 #{done}] {sample['n_questions']} 个问题")

    with open(jsonl_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\n=== 完成：{len(samples)} 条样本 → {output_filename}（+ {jsonl_path}）；降级 {degraded} 次 ===")
    return samples

# ---------------------------------------------------------------------------
# 7. 执行入口
# ---------------------------------------------------------------------------
def load_domain_config(domain: str, config_dir: str = ".") -> dict:
    """加载领域配置文件（关键词、提示语模板等）。"""
    import json as _json
    import os as _os

    # 尝试从 config_{domain}.json 加载
    for path in [
        _os.path.join(config_dir, f"config_{domain}.json"),
        _os.path.join(config_dir, f"config_{domain.lower()}.json"),
        "config.json",
    ]:
        if _os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = _json.load(f)
            if "keywords" in cfg and "state_prompt_template" in cfg:
                # 确保有 doc_name 字段
                if "doc_name" not in cfg:
                    cfg["doc_name"] = domain_doc_names.get(domain.lower(), "政策文档")
                return cfg

    # 默认配置
    domain_doc_names = {
        "compliance": "合规手册",
        "healthcare": "医疗政策文档",
        "education": "教育政策文档",
        "e-commerce": "电商平台规则",
        "hr": "人力资源政策文档",
    }
    default_config = {
        "domain": domain,
        "doc_name": domain_doc_names.get(domain.lower(), "政策文档"),
        "keywords": ["Risk", "Compliance", "Fraud", "Privacy", "Record",
                     "Account", "Transaction", "Employee", "Performance", "Termination"],
        "state_prompt_template": (
            f"你是一个专业的{domain_doc_names.get(domain.lower(), '业务管理')}测试数据生成器。根据以下{domain_doc_names.get(domain.lower(), '相关政策文档')}，生成 {{batch_size}} 个不同维度的虚拟申请/事件状态。\n\n"
            "要求：\n"
            "1. 必须包含常规案例、明显违规案例，以及大量处于灰色地带的边界案例（Edge Cases）。\n"
            "2. 场景应覆盖该领域中的多个关键决策点：风险管理、合规流程、交易执行、隐私保护等。\n"
            "3. 返回格式必须是纯 JSON 对象，包含一个 states 键，对应一个由多个状态字典组成的列表。\n"
            "4. 每个状态字典应包含：state_id（唯一标识）、domain（领域）、description（详细描述）、client_info（客户信息）、transaction_info（交易信息）、risk_indicators（风险指标列表）。\n\n"
            "【{{doc_name}}】\n{{policy_text}}"
        ),
    }
    return default_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用合规手册生成决策模型训练数据集")
    parser.add_argument("--pdf", required=True, help="政策文档PDF路径")
    parser.add_argument("--txt", default=None, help="可选：预提取的文本文件路径")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--output", default="dataset.json")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--domain", required=True, help="领域：compliance, healthcare, education, e-commerce, hr")
    parser.add_argument("--config-dir", default=".", help="配置文件目录")
    parser.add_argument("--batch-size", type=int, default=10, help="每次状态生成数量")
    parser.add_argument("--questions-per-sample", type=int, default=3, help="每个样本的问题数")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="最大token数（默认 16384；模型消耗 reasoning tokens，state 生成需要较大值）")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发 worker 数（测试环境 LLM 不支持并发时保持 1）")
    parser.add_argument("--ensemble-n", type=int, default=1,
                        help="teacher 采样次数取均值（3 = 参考文件的 x3 mean）")
    parser.add_argument("--resume", action="store_true",
                        help="从已有 JSONL 继续生成到 target")
    parser.add_argument("--templates-dir", default=".", help="模板目录（含 templates/ 子目录）")
    args = parser.parse_args()

    # 使用用户指定的PDF（不自动选择）
    txt_path = args.txt if args.txt else os.path.splitext(args.pdf)[0] + ".txt"
    args.txt = txt_path

    # 加载领域配置
    domain_config = load_domain_config(args.domain, config_dir=args.config_dir)
    print(f"[领域配置] domain={args.domain}, keywords={len(domain_config.get('keywords', []))}")

    policy_text = extract_policy_text(args.pdf, args.txt)
    # 使用配置的关键词提取相关章节
    policy_text = extract_relevant_sections(policy_text, max_chars=30000,
                                            keywords=domain_config.get("keywords"))
    print(f"[策略文档] 已加载 {len(policy_text)} 字符")

    config = load_model_config(args.config)
    print(f"[模型配置]")
    for key in ["state_model", "teacher_model", "gold_model"]:
        m = config[key]
        print(f"  {key}: {m['name']} @ {m['base_url']}")

    state_llm = LLMClient(
        api_key=config["state_model"]["api_key"],
        base_url=config["state_model"]["base_url"],
        model=config["state_model"]["name"],
    )
    teacher_llm = LLMClient(
        api_key=config["teacher_model"]["api_key"],
        base_url=config["teacher_model"]["base_url"],
        model=config["teacher_model"]["name"],
    )
    gold_llm = LLMClient(
        api_key=config["gold_model"]["api_key"],
        base_url=config["gold_model"]["base_url"],
        model=config["gold_model"]["name"],
    )

    generate_dataset(state_llm, teacher_llm, policy_text, domain=args.domain,
                     target_samples=args.target, output_filename=args.output,
                     batch_size=args.batch_size, max_tokens=args.max_tokens,
                     questions_per_sample=args.questions_per_sample,
                     concurrency=args.concurrency, ensemble_n=args.ensemble_n,
                     resume=args.resume, templates_dir=args.templates_dir)
