#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一步：根据政策文档生成配置文件。
分析文档内容，提取关键章节、生成相关关键词和提示语，输出到配置文件。
"""

import json
import os
import re
import sys

from llm_client import LLMClient
from template_store import load_templates, merge_templates, save_templates


def extract_document_structure(text: str, max_sections: int = 50) -> list:
    """提取文档的章节结构。"""
    lines = text.split("\n")
    sections = []
    current_title = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        is_heading = (
            stripped and len(stripped) < 120
            and stripped[0].isupper()
            and not stripped.startswith("•")
            and not stripped.startswith("-")
            and "\t" not in stripped
            and len(stripped) > 3
        )

        if is_heading:
            if current_title and current_lines:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines[:200]),
                    "length": sum(len(l) for l in current_lines),
                })
            current_title = stripped
            current_lines = []
        else:
            if current_title:
                current_lines.append(line)

    if current_title and current_lines:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines[:200]),
            "length": sum(len(l) for l in current_lines),
        })

    # 按长度排序，取最大的几个
    sections.sort(key=lambda s: s["length"], reverse=True)
    return sections[:max_sections]


def suggest_keywords(sections: list, domain: str = None) -> list:
    """根据文档章节和领域建议相关关键词。"""
    # 领域特定关键词
    domain_keywords = {
        "compliance": ["Fiduciary", "Compliance Risk", "Registration", "Licensing",
                       "Anti Money Laundering", "KYC", "Customer Due Diligence",
                       "Books and Records", "Custody", "Privacy", "Cybersecurity",
                       "Fraudulent Transfer", "Identity Theft", "Business Continuity",
                       "Best Execution", "Suitability", "Account Opening",
                       "Proxy Voting", "Advertising", "Social Networking",
                       "Correspondence", "Email", "Complaints", "Disputes",
                       "Diminished Capacity", "Vulnerable Client",
                       "ERISA", "Prohibited Transaction", "Conflicts of Interest",
                       "Form ADV", "Disclosure", "Client Contract"],
        "healthcare": ["Patient Safety", "Medical Record", "Informed Consent",
                       "Treatment", "Prescription", "Drug Interaction",
                       "Infection Control", "Emergency", "Discharge Planning",
                       "Special Needs", "Accommodation", "Privacy", "HIPAA",
                       "Clinical Guideline", "Standard of Care", "Malpractice",
                       "Patient Rights", "Advance Directive", "End of Life"],
        "education": ["Plagiarism", "Academic Integrity", "Cheating",
                      "Assignment", "Submission", "Grading", "Evaluation",
                      "Attendance", "Behavior", "Discipline",
                      "Special Education", "IEP", "504 Plan",
                      "Student Rights", "Privacy", "FERPA",
                      "Learning Outcome", "Assessment", "Curriculum",
                      "Teacher Conduct", "Harassment", "Bullying"],
        "e-commerce": ["Product Listing", "Listing Compliance", "Prohibited Item",
                       "Return Policy", "Refund", "Cancellation",
                       "Payment Fraud", "Chargeback", "Money Laundering",
                       "Identity Verification", "KYC", "AML",
                       "Shipping", "Delivery", "Logistics",
                       "Customer Service", "Complaint", "Dispute",
                       "Review", "Rating", "Fake Review",
                       "Data Privacy", "GDPR", "CCPA",
                       "Consumer Protection", "Warranty", "Guarantee"],
        "hr": ["Employee Relations", "Performance Management", "Termination",
               "Hiring", "Onboarding", "Benefits", "Compensation",
               "Labor Law", "Discrimination", "Harassment",
               "Workplace Safety", "Training", "Attendance", "Leave Management",
               "Job Description", "Recruitment", "Selection", "Promotion",
               "Severance", "Wrongful Termination", "Workers Compensation",
               "Overtime", "Minimum Wage", "Collective Bargaining",
               "Union", "Grievance", "Mediation", "Arbitration",
               "Employee Handbook", "Code of Conduct", "Confidentiality",
               "Non-Disclosure", "Non-Compete", "Intellectual Property",
               "Remote Work", "Flexible Schedule", "Work-Life Balance",
               "Mental Health", "EAP", "Diversity", "Inclusion"],
    }

    if domain and domain.lower() in domain_keywords:
        return domain_keywords[domain.lower()]

    # 通用关键词（所有领域都适用）
    return ["Risk", "Compliance", "Violation", "Policy", "Procedure",
            "Approval", "Authorization", "Verification", "Validation",
            "Audit", "Record", "Documentation", "Disclosure",
            "Conflict of Interest", "Fraud", "Misconduct",
            "Breach", "Incident", "Report", "Investigation",
            "Remediation", "Corrective Action", "Prevention"]


def generate_state_prompt(domain: str, sections: list) -> str:
    """生成领域特定的状态提示语。"""
    domain_names = {
        "compliance": ("合规与风险管理", "合规手册"),
        "healthcare": ("医疗健康", "医疗政策文档"),
        "education": ("教育管理", "教育政策文档"),
        "e-commerce": ("电子商务", "电商平台规则"),
        "hr": ("人力资源管理", "人力资源政策文档"),
    }
    domain_name, doc_name = domain_names.get(domain.lower(), ("业务管理", "相关政策文档"))

    # 从文档中提取关键章节作为场景覆盖建议（基于领域关键词）
    domain_keywords_for_extraction = {
        "compliance": ["Risk", "Compliance", "Fraud", "Privacy", "Record", "Account",
                       "Transaction", "Custody", "Fiduciary", "Audit"],
        "healthcare": ["Patient", "Medical", "Treatment", "Prescription", "Infection",
                       "Safety", "Emergency", "Clinical", "Standard of Care", "Malpractice"],
        "education": ["Plagiarism", "Academic", "Assignment", "Grading", "Attendance",
                      "Behavior", "Special Education", "Student Rights", "Assessment", "Curriculum"],
        "e-commerce": ["Product", "Listing", "Return", "Payment", "Shipping", "Review",
                       "Fraud", "Consumer", "Warranty", "Dispute"],
        "hr": ["Employee", "Performance", "Termination", "Hiring", "Benefits",
               "Labor", "Training", "Discrimination", "Harassment", "Compensation"],
    }
    extraction_keywords = domain_keywords_for_extraction.get(domain.lower(),
                                                             ["Risk", "Compliance", "Fraud", "Privacy",
                                                              "Record", "Account", "Transaction"])

    key_sections = []
    for s in sections[:20]:
        title = s["title"].strip()
        if any(kw.lower() in title.lower() for kw in extraction_keywords):
            key_sections.append(title)

    # 领域特定的默认决策点（当文档中没有匹配章节时）
    default_decision_points = {
        "compliance": "风险管理、合规流程、交易执行、隐私保护",
        "healthcare": "患者安全、治疗方案、病历管理、用药风险",
        "education": "学术诚信、作业评估、学生行为、学习成果",
        "e-commerce": "商品发布、退货退款、支付安全、客户服务",
        "hr": "员工关系、绩效管理、招聘入职、薪酬福利",
    }
    section_list = ", ".join(key_sections[:8]) if key_sections else default_decision_points.get(domain.lower(), "风险管理、合规流程、交易执行、隐私保护")

    # 领域特定的状态字段说明
    domain_field_names = {
        "compliance": "client_info（客户信息）、transaction_info（交易信息）",
        "healthcare": "patient_info（患者信息）、treatment_info（治疗信息）",
        "education": "student_info（学生信息）、academic_info（学业信息）",
        "e-commerce": "customer_info（顾客信息）、order_info（订单信息）",
        "hr": "employee_info（员工信息）、employment_info（雇佣信息）",
    }
    extra_fields = domain_field_names.get(domain.lower(), "client_info（客户信息）、transaction_info（交易信息）")

    prompt = (
        f"你是一个专业的{domain_name}测试数据生成器。根据以下{doc_name}，生成 {{batch_size}} 个不同维度的虚拟申请/事件状态。\n\n"
        f"要求：\n"
        f"1. 必须包含常规案例、明显违规案例，以及大量处于灰色地带的边界案例（Edge Cases）。\n"
        f"2. 场景应覆盖该领域中的多个关键决策点：{section_list}等。\n"
        f"3. 返回格式必须是纯 JSON 对象，包含一个 states 键，对应一个由多个状态字典组成的列表。\n"
        f"4. 每个状态字典应包含：state_id（唯一标识）、domain（领域）、description（详细描述）、{extra_fields}risk_indicators（风险指标列表）。\n\n"
        f"【{doc_name}】\n{{policy_text}}"
    )

    return prompt


RULE_GRAMMAR = """facts 规格（rule.op 选其一）：
- identity: gold = state[field]（choice；field 值必须是 criteria 的键）
- map: gold = mapping[state[field]]（score 输出整数索引；choice 输出键）
- count_gte: true 当 len(state[field]) >= value（noul）
- gte / lte / eq / in: 数值或成员比较（noul）"""


def build_template_generation_prompt(domain, doc_name, policy_text, existing_keys):
    return (
        f"你是{doc_name}分析器。基于以下政策文档，为领域 {domain} 生成决策问题模板。\n\n"
        "要求：\n"
        "1. 生成 8-15 个模板，覆盖文档中的关键决策点；类型混合 choice/noul/score（每种至少 2 个）。\n"
        "2. 每个模板必须包含 facts 规格，用于从 state 字段确定性推导答案：\n"
        f"{RULE_GRAMMAR}\n"
        f"3. key 不得与以下已有 key 重复：{', '.join(existing_keys) if existing_keys else '（无）'}\n"
        "4. policy_keywords: 2-4 个用于定位文档相关章节的英文关键词。\n"
        "5. 模板示例（facts 结构必须严格如下，field 在 facts 顶层、不在 rule 内）：\n"
        '{"key": "example_q", "type": "noul", "instructions": "...", '
        '"criteria": {"true": "...", "false": "..."}, '
        '"facts": {"field": "risk_flags", "rule": {"op": "count_gte", "value": 2}}, '
        '"policy_keywords": ["Keyword A", "Keyword B"]}\n'
        "6. 只返回 JSON：{\"decision_points\": [...], \"templates\": [{key, type, instructions, criteria, facts, policy_keywords}]}\n\n"
        f"【政策文档】\n{policy_text}"
    )


def filter_valid_templates(data, existing_keys):
    """过滤 LLM 返回的模板：字段齐全、type 合法、facts 完整、key 不重复。"""
    templates = data.get("templates", []) if isinstance(data, dict) else []
    valid = []
    for t in templates:
        if not isinstance(t, dict):
            print(f"  [模板拒绝] 非字典: {str(t)[:80]}")
            continue
        missing = [k for k in ("key", "type", "instructions", "criteria", "facts") if k not in t]
        if missing:
            print(f"  [模板拒绝] 缺字段 {missing}: key={t.get('key')!r}")
            continue
        if t["type"] not in ("choice", "noul", "score"):
            print(f"  [模板拒绝] 类型非法 {t['type']!r}: key={t['key']!r}")
            continue
        if t["key"] in existing_keys:
            print(f"  [模板拒绝] key 重复: {t['key']!r}")
            continue
        facts = t["facts"]
        if not (isinstance(facts, dict) and isinstance(facts.get("rule"), dict)
                and "op" in facts["rule"]):
            print(f"  [模板拒绝] facts 规格不完整: key={t['key']!r}, facts={str(facts)[:120]}")
            continue
        # 兼容模型把 field 放在 rule 内的写法：提升到顶层
        if "field" not in facts and isinstance(facts["rule"].get("field"), str):
            facts = dict(facts)
            facts["field"] = facts["rule"].pop("field")
        if "field" not in facts:
            print(f"  [模板拒绝] facts 缺 field: key={t['key']!r}, facts={str(facts)[:120]}")
            continue
        t = dict(t)
        t["facts"] = facts
        valid.append(t)
    return valid


def generate_templates_from_document(llm, policy_text, domain, doc_name, existing_keys):
    """调用 LLM 从文档生成新模板。返回 (decision_points, valid_templates)。"""
    prompt = build_template_generation_prompt(domain, doc_name, policy_text, existing_keys)
    data = llm.chat_json(messages=[{"role": "user", "content": prompt}],
                         temperature=0.7, max_tokens=8192)
    points = data.get("decision_points", []) if isinstance(data, dict) else []
    return points, filter_valid_templates(data, existing_keys)


def generate_config(policy_text: str, domain: str = "compliance", output_dir: str = ".") -> dict:
    """生成完整的配置文件。"""
    sections = extract_document_structure(policy_text)
    keywords = suggest_keywords(sections, domain)
    state_prompt_template = generate_state_prompt(domain, sections)

    # 根据领域关键词过滤文档章节，生成相关的摘要
    domain_keywords_for_extraction = {
        "compliance": ["Risk", "Compliance", "Fraud", "Privacy", "Record", "Account",
                       "Transaction", "Custody", "Fiduciary", "Audit"],
        "healthcare": ["Patient", "Medical", "Treatment", "Prescription", "Infection",
                       "Safety", "Emergency", "Clinical", "Standard of Care", "Malpractice"],
        "education": ["Plagiarism", "Academic", "Assignment", "Grading", "Attendance",
                      "Behavior", "Special Education", "Student Rights", "Assessment", "Curriculum"],
        "e-commerce": ["Product", "Listing", "Return", "Payment", "Shipping", "Review",
                       "Fraud", "Consumer", "Warranty", "Dispute"],
        "hr": ["Employee", "Performance", "Termination", "Hiring", "Benefits",
               "Labor", "Training", "Discrimination", "Harassment", "Compensation"],
    }
    extraction_keywords = domain_keywords_for_extraction.get(domain.lower(),
                                                             ["Risk", "Compliance", "Fraud", "Privacy",
                                                              "Record", "Account", "Transaction"])

    relevant_sections = []
    for s in sections[:50]:
        title = s["title"].strip()
        if any(kw.lower() in title.lower() for kw in extraction_keywords):
            relevant_sections.append(title)

    # 如果找到相关章节，使用它们；否则使用全部章节
    summary_sections = relevant_sections[:10] if relevant_sections else [s["title"] for s in sections[:10]]

    # 领域特定的文档名称
    domain_doc_names = {
        "compliance": "合规手册",
        "healthcare": "医疗政策文档",
        "education": "教育政策文档",
        "e-commerce": "电商平台规则",
        "hr": "人力资源政策文档",
    }
    doc_name = domain_doc_names.get(domain.lower(), "政策文档")

    config = {
        "domain": domain,
        "doc_name": doc_name,
        "keywords": keywords,
        "state_prompt_template": state_prompt_template,
        "document_summary": {
            "total_chars": len(policy_text),
            "num_sections": len(sections),
            "top_sections": summary_sections,
            "relevant_to_domain": len(relevant_sections) > 0,
            "domain_note": f"从文档中提取了 {len(relevant_sections)} 个与{domain}领域相关的章节" if relevant_sections else f"文档中未找到明显的{domain}相关章节，建议使用{domain}领域的政策文档"},
    }

    # 保存配置文件
    config_path = os.path.join(output_dir, f"config_{domain}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[配置文件] 已生成 {config_path}")
    print(f"  领域: {domain}")
    print(f"  关键词数量: {len(keywords)}")
    print(f"  文档章节数: {len(sections)}")
    if sections:
        print(f"  主要章节: {', '.join(s['title'] for s in sections[:5])}")

    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="根据政策文档生成配置文件")
    parser.add_argument("--pdf", required=True, help="政策文档PDF路径")
    parser.add_argument("--txt", default=None, help="可选：预提取的文本文件路径")
    parser.add_argument("--domain", required=True, help="领域：compliance, healthcare, education, e-commerce, hr")
    parser.add_argument("--output-dir", default=".", help="输出目录")
    parser.add_argument("--skip-templates", action="store_true",
                        help="跳过 LLM 模板生成，仅生成配置文件")
    parser.add_argument("--config", default="config.json", help="模型配置路径")
    args = parser.parse_args()

    # 提取文本：始终从 PDF 提取并保存为 txt，确保内容最新
    try:
        from pdfminer.high_level import extract_text
        txt_path = args.txt if args.txt else os.path.splitext(args.pdf)[0] + ".txt"
        text = extract_text(args.pdf)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[PDF 提取] 已保存 {len(text)} 字符到 {txt_path}")
    except ImportError:
        # 如果 pdfminer 不可用，尝试读取现有 txt 文件
        if args.txt and os.path.exists(args.txt):
            with open(args.txt, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"[文本文件] 使用现有文件 {args.txt} ({len(text)} 字符)")
        else:
            print("需要 pdfminer.six: pip install pdfminer.six")
            sys.exit(1)

    print(f"[文档] 已加载 {len(text)} 字符")

    generate_config(text, domain=args.domain, output_dir=args.output_dir)

    # --- 文档驱动模板生成 + 合并（可用 --skip-templates 跳过）---
    if not args.skip_templates:
        from config_loader import load_model_config
        cfg = load_model_config(args.config)
        llm = LLMClient(api_key=cfg["state_model"]["api_key"],
                        base_url=cfg["state_model"]["base_url"],
                        model=cfg["state_model"]["name"])
        existing = load_templates(args.domain, base=".")
        existing_keys = {t.get("key") for t in existing}
        doc_name = {"compliance": "合规手册", "healthcare": "医疗政策文档",
                    "education": "教育政策文档", "e-commerce": "电商平台规则",
                    "hr": "人力资源政策文档"}.get(args.domain.lower(), "政策文档")
        points, new_templates = generate_templates_from_document(
            llm, text, args.domain, doc_name, existing_keys)
        merged, added = merge_templates(existing, new_templates)
        path = save_templates(merged, args.domain, base=".")
        print(f"[模板生成] 新增 {added} 个模板 → {path}（共 {len(merged)} 个）")
        if points:
            print(f"[决策点] {', '.join(points[:10])}")
