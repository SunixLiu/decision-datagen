# 决策型模型训练数据集生成程序

基于用户提供的政策文档（PDF，如合规手册、员工手册），生成 Jev 三原语（choice/noul/score）类型的结构化决策训练数据。

## 目标格式

参考文件：`sql-console-for-n4ze3m-typed-decisions-synth.json`

每条样本包含：
- `state_id`: 唯一标识
- `generator`: 生成器版本
- `domain`: 领域名称
- `n_questions`: 问题数量
- `state`: JSON 字符串格式的状态描述
- `state_is_json`: 是否为 JSON
- `questions`: 问题定义（type/criteria/instructions）
- `gold`: 金标准答案（deterministic）
- `teacher`: 教师模型概率化预测
- `teacher_model`: 教师模型名称

### Jev 三原语类型

| 类型 | 说明 | Gold 格式 | Teacher 格式 |
|------|------|-----------|--------------|
| choice | 多选一决策 | 选项名称 | `{"probabilities": {"option1": 0.7, ...}}` |
| noul | 是非判断 | true/false | `{"noul": 0.8}` |
| score | 等级评分 | 索引/名称 | `{"probabilities": {"low": 0.2, ...}}` |

## 输出格式

- `questions` / `gold` / `teacher` 均为 **JSON 字符串**（非嵌套对象），需 `json.loads` 解析。
- score 类型 gold：criteria 列表的整数索引（0 起）。
- score 类型 teacher：probabilities 以索引字符串为键（`"0"`、`"1"`、...），总和归一化为 1.0。
- choice 类型 teacher：probabilities 以选项名为键；noul 类型 teacher：`{"noul": float}`。
- `generator`: `v1-{domain}`（如 `v1-compliance`）。
- 双输出：JSON 数组文件（如 `dataset_50.json`）+ 增量 JSONL（`dataset_50.jsonl`，每行一条样本，用于 resume）。

## 三步流程

### 第一步：生成配置文件与问题模板

使用 `generate_config.py` 根据政策文档生成领域配置文件，并调用 LLM 一次生成文档驱动的问题模板，追加到 `templates/{domain}.json`（按 key 去重）：

```bash
python generate_config.py --pdf <政策文档路径>.pdf --domain compliance
```

输出：
- `config_compliance.json`：包含 `keywords`（用于提取相关章节的关键词列表）、`state_prompt_template`（状态生成提示语模板）、`document_summary`（文档摘要信息）等
- `templates/compliance.json`：追加新生成的问题模板（带 facts 规格，按 key 去重）

离线模式（跳过 LLM 模板生成，仅生成配置文件）：

```bash
python generate_config.py --pdf <政策文档路径>.pdf --domain compliance --skip-templates
```

### 第二步：生成数据集

使用 `generate_dataset.py` 加载配置文件并生成训练数据：

```bash
python generate_dataset.py --pdf <政策文档路径>.pdf --domain compliance --target 50 --output dataset_50.json [--resume] [--concurrency N] [--ensemble-n N]
```

- `--concurrency`: 并发 worker 数，默认 1（测试环境 LLM 仅支持顺序调用，保持 1）
- `--ensemble-n`: teacher 采样次数取均值；3 = 参考文件的 "x3 mean"
- `--resume`: 从已有 JSONL 继续生成到 target（已完成样本不重复生成）

### 第三步：评估与转换

```bash
python evaluate_dataset.py --input dataset_50.json
python convert_for_finetune.py --input dataset_50.json --output finetune_data.jsonl
```

## 三模型架构

程序使用三个独立配置的 LLM：

| 模型 | 用途 | 配置项 |
|------|------|--------|
| state_model | 生成场景状态 | `state_model.name`, `state_model.api_key`, `state_model.base_url` |
| teacher_model | 提供概率化预测 | `teacher_model.name`, `teacher_model.api_key`, `teacher_model.base_url` |
| gold_model | 确定金标准答案 | `gold_model.name`, `gold_model.api_key`, `gold_model.base_url` |

配置文件见 `config.json`，支持环境变量覆盖（`STATE_MODEL_NAME`, `TEACHER_API_KEY`, `GOLD_BASE_URL` 等）。

## 三模型使用建议

- **state_model**（场景状态生成）：可用较小的模型——只需按提示词产出结构化场景状态。
- **teacher_model / gold_model**（概率化预测与金标准）：实际使用中建议使用更大/更强的模型，以保证预测质量与答案可靠性。
- `--ensemble-n 3` 对应参考文件的 "x3 mean"：采样 3 次取概率均值。

## 多领域支持

内置 60+ 个 Jev 三原语问题模板，覆盖多个领域：

| 领域 | 模板数 | 关键主题 |
|------|--------|----------|
| compliance | 23 | 账户开立、反洗钱、交易执行、隐私、欺诈、投诉、记录、认知能力、广告、ERISA、注册、业务连续性 |
| healthcare | 8 | 患者 consent、治疗方案、病历、处方风险、感染控制、患者安全、急诊、出院计划 |
| education | 8 | 抄袭、作业合规、学术诚信、评分公平、特殊需求、出勤、行为、学习成果 |
| e-commerce | 8 | 商品发布、退货、卖家合规、评价真实性、支付欺诈、发货、退款、纠纷 |
| hr | 18 | 入职合规、绩效评估、骚扰投诉、请假审批、薪酬合规、离职程序、培训、行为准则、远程办公、数据隐私、加班、职场安全、招聘偏见、晋升公平、福利、员工敬业度、利益冲突、纪律处分 |

## 程序结构

| 文件 | 说明 |
|------|------|
| `generate_dataset.py` | 主程序：三模型架构、多领域支持、配置驱动 |
| `generate_config.py` | 第一步：从政策文档生成配置文件 + 文档驱动问题模板 |
| `evaluate_dataset.py` | 评估数据集质量（Schema 校验、类型分布、gold/teacher 一致率） |
| `convert_for_finetune.py` | 转换为 OpenAI fine-tuning JSONL 格式 |
| `llm_client.py` | OpenAI 兼容 LLM 客户端（chat/chat_json，重试与 JSON 提取） |
| `template_store.py` | 按领域加载/保存问题模板（`templates/{domain}.json`，按 key 去重合并） |
| `gold_rules.py` | 从 state 内置事实确定性推导 gold（identity/map/count_gte/gte/lte/eq/in） |
| `teacher.py` | teacher 概率预测的校验、归一化与 ensemble 均值 |
| `config_loader.py` | 三模型端点配置加载（config.json + 环境变量覆盖） |
| `templates/{domain}.json` | 各领域问题模板（**当前唯一事实来源**；compliance 23 个等） |
| `question_templates.json` | 迁移前备份（旧格式），不再被程序读取 |
| `scripts/split_templates.py` | 将旧 `question_templates.json` 拆分为 `templates/` 各领域文件 |
| `tests/` | 单元测试（gold_rules、template_store、teacher、Schema 校验等） |
| `config.json` | 三个独立 LLM 端点配置 |

## 使用示例

```bash
# 1. 生成配置文件 + 文档驱动模板
python generate_config.py --pdf <政策文档路径>.pdf --domain compliance

# 2. 生成数据集
python generate_dataset.py --pdf <政策文档路径>.pdf --target 10 --domain compliance --output dataset_10.json

# 3. 评估数据集
python evaluate_dataset.py --input dataset_10.json

# 4. 转换为 fine-tuning 格式
python convert_for_finetune.py --input dataset_10.json --output finetune_data.jsonl
```

## 输出示例

```json
{
  "state_id": "synth-xxx",
  "generator": "v1-compliance",
  "domain": "compliance",
  "n_questions": 3,
  "state": "{\"description\": \"...\", \"client_info\": {...}, ...}",
  "state_is_json": true,
  "questions": "{\"account_compliant\": {\"type\": \"choice\", \"instructions\": \"判断该账户开立流程是否符合公司合规政策。\", \"criteria\": {\"compliant\": \"...\", \"non-compliant\": \"...\", \"needs-review\": \"...\"}}}",
  "gold": "{\"account_compliant\": \"compliant\"}",
  "teacher": "{\"account_compliant\": {\"probabilities\": {\"compliant\": 0.95, \"non-compliant\": 0.03, \"needs-review\": 0.02}}}",
  "teacher_model": "Qwen3.6-35B-A3B"
}
```

注意：`questions` / `gold` / `teacher` 是 **JSON 字符串**（如上所示），消费方需 `json.loads` 解析。

### 如何新增领域

要添加新领域（例如 "finance"），需要修改以下文件：

1. **`templates/finance.json`** - 新建该领域的 Jev 三原语问题模板列表（每个模板必须带 `facts` 规格，否则不参与问题选择）：
   ```json
   [
     {
       "key": "loan_approval",
       "type": "choice",
       "instructions": "判断贷款申请是否符合审批标准。",
       "criteria": {"approved": "...", "rejected": "...", "needs-review": "..."},
       "facts": {"field": "loan.status", "rule": {"op": "identity"}},
       "policy_keywords": ["Loan Approval", "Credit"]
     }
   ]
   ```

2. **`generate_config.py`** - 在以下几个地方添加领域特定信息：
   - `domain_keywords` 字典：添加该领域的关键词列表
   - `domain_field_names` 字典：添加状态字典的字段名称
   - `default_decision_points` 字典：添加默认决策点
   - `domain_names` 字典：添加领域显示名称和文档名称
   - `domain_keywords_for_extraction` 字典：添加用于章节提取的关键词

3. **生成配置并测试**：
   ```bash
   python generate_config.py --domain finance --pdf <财务手册路径>.pdf
   python generate_dataset.py --target 10 --domain finance --output dataset_finance_10.json
   ```

# 注意事项

1. 生成大量样本时需要较长时间，建议分批运行（每批 5-10 条）
2. 确保 LLM 服务器正常运行且模型可用
3. 生成的数据仅用于训练目的，需人工审核后再用于生产
4. 程序默认提取政策文档中与决策最相关的章节（约 30K 字符）
5. 每个样本包含 3-4 个随机选择的问题，覆盖不同领域
6. 教师模型使用 Qwen3.6-35B-A3B，temperature=0.3 以获取概率化预测
7. gold 由规则从 state 内置事实确定性推导；某问题无法推导时跳过该问题，若整条样本无可推导问题则丢弃并计入降级次数
8. `--resume` 可从已有 JSONL 继续生成到 target（已完成样本不重复生成）
9. 状态生成连续失败 3 次时程序中止，已完成部分保留在 JSONL 中，可用 `--resume` 续跑
10. 无法产出任何问题的样本会计入"降级"次数（完成行会报告降级次数），不写入数据集
11. 不带 facts 规格的模板不参与问题选择（仅带 facts 的模板进入采样池）

## 环境要求

- Python 3.8+
- requests
- pdfminer.six（用于 PDF 文本提取）

```bash
pip install requests pdfminer.six
```
