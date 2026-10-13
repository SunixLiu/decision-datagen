# Decision Model Training Dataset Generator

Generates structured decision training data in the Jev three-primitive format (choice/noul/score) from a user-supplied policy document (PDF, e.g. a compliance manual or employee handbook).

中文文档: [README_zh.md](README_zh.md)

## Target Format

Reference file: `sql-console-for-n4ze3m-typed-decisions-synth.json`

Each sample contains:
- `state_id`: unique identifier
- `generator`: generator version
- `domain`: domain name
- `n_questions`: number of questions
- `state`: state description as a JSON string
- `state_is_json`: whether the state is JSON
- `questions`: question definitions (type/criteria/instructions)
- `gold`: deterministic gold answer
- `teacher`: teacher model probabilistic prediction
- `teacher_model`: teacher model name

### Jev Three Primitive Types

| Type | Description | Gold format | Teacher format |
|------|-------------|-------------|----------------|
| choice | Pick-one decision | option name | `{"probabilities": {"option1": 0.7, ...}}` |
| noul | yes/no judgment | true/false | `{"noul": 0.8}` |
| score | ordinal rating | index/name | `{"probabilities": {"low": 0.2, ...}}` |

## Output Format

- `questions` / `gold` / `teacher` are all **JSON strings** (not nested objects); parse with `json.loads`.
- score gold: integer index into the criteria list (0-based).
- score teacher: probabilities keyed by index strings (`"0"`, `"1"`, ...), normalized to sum to 1.0.
- choice teacher: probabilities keyed by option name; noul teacher: `{"noul": float}`.
- `generator`: `v1-{domain}` (e.g. `v1-compliance`).
- Dual output: JSON array file (e.g. `dataset_50.json`) + incremental JSONL (`dataset_50.jsonl`, one sample per line, used for resume).
- The output is already the target fine-tuning data format (field-aligned with the reference file); training pipelines that consume this format can use it directly, no extra conversion needed.

## Target Model Type & Fine-tuning Integration

This dataset targets **System One / Jev-style decision models** (encoder + decision head architecture, e.g. the open-source [Laya](https://github.com/datawhalechina/jev-cookbook)) — **not conversational SFT data**: such models take a state + typed questions and directly output probability distributions, with no text generation and no parsing.

Field mapping to the training schema (see [Jev Cookbook Chapter 10 FINETUNING.md](https://github.com/datawhalechina/jev-cookbook/blob/main/main/10_%E6%9C%AC%E5%9C%B0%E6%A8%A1%E5%9E%8B/FINETUNING.md)):

| Dataset field | Training schema | Notes |
|---|---|---|
| `state` | `state` | state visible at decision time |
| `questions` | `qs` | typed questions (choice/noul/score); one question asks exactly one thing |
| `gold` | `y` | deterministic label (choice=option name, score=candidate index, noul=bool); the trainer maps it to index/0-1 per its own schema |
| `teacher` | `soft` | soft target probability distribution, used for masked soft cross-entropy training |

Public benchmark dataset with the same schema: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (one state answered by multiple typed questions at once; every answer is a probability distribution).

## Three-Step Workflow

### Step 1: Generate Config File and Question Templates

Use `generate_config.py` to generate the domain config from the policy document, and call the LLM once to generate document-driven question templates, appended to `templates/{domain}.json` (deduplicated by key):

```bash
python generate_config.py --pdf <path-to-policy-doc>.pdf --domain compliance
```

Outputs:
- `config_compliance.json`: contains `keywords` (keyword list for extracting relevant sections), `state_prompt_template` (state generation prompt template), `document_summary` (document summary), etc.
- `templates/compliance.json`: newly generated question templates appended (with facts specs, deduplicated by key)

Offline mode (skip LLM template generation, config only):

```bash
python generate_config.py --pdf <path-to-policy-doc>.pdf --domain compliance --skip-templates
```

### Step 2: Generate the Dataset

Use `generate_dataset.py` to load the config and generate training data:

```bash
python generate_dataset.py --pdf <path-to-policy-doc>.pdf --domain compliance --target 50 --output dataset_50.json [--resume] [--concurrency N] [--ensemble-n N]
```

- `--concurrency`: number of concurrent workers, default 1 (keep at 1 if your LLM server only supports sequential calls)
- `--ensemble-n`: number of teacher samples averaged; 3 = the reference file's "x3 mean"
- `--resume`: continue from an existing JSONL up to target (already-completed samples are not regenerated)

### Step 3: Evaluate

```bash
python evaluate_dataset.py --input dataset_50.json
```

## Three-Model Architecture

The program uses three independently configured LLMs:

| Model | Purpose | Config keys |
|-------|---------|-------------|
| state_model | generates scenario states | `state_model.name`, `state_model.api_key`, `state_model.base_url` |
| teacher_model | provides probabilistic predictions | `teacher_model.name`, `teacher_model.api_key`, `teacher_model.base_url` |
| gold_model | determines gold answers | `gold_model.name`, `gold_model.api_key`, `gold_model.base_url` |

See `config.json`; environment variable overrides are supported (`STATE_MODEL_NAME`, `TEACHER_API_KEY`, `GOLD_BASE_URL`, etc.).

## Model Usage Recommendations

- **state_model** (scenario state generation): a smaller model is fine — it only needs to produce structured scenario states from the prompt.
- **teacher_model / gold_model** (probabilistic prediction and gold answers): use a larger/stronger model in practice, to guarantee prediction quality and answer reliability.
- `--ensemble-n 3` corresponds to the reference file's "x3 mean": sample 3 times and average the probabilities.

## Multi-Domain Support

60+ built-in Jev three-primitive question templates across multiple domains:

| Domain | Templates | Key topics |
|--------|-----------|------------|
| compliance | 23 | account opening, AML, trade execution, privacy, fraud, complaints, records, cognitive ability, advertising, ERISA, registration, business continuity |
| healthcare | 8 | patient consent, treatment plans, medical records, prescription risk, infection control, patient safety, emergency care, discharge planning |
| education | 8 | plagiarism, assignment compliance, academic integrity, grading fairness, special needs, attendance, behavior, learning outcomes |
| e-commerce | 8 | product listing, returns, seller compliance, review authenticity, payment fraud, shipping, refunds, disputes |
| hr | 18 | onboarding compliance, performance review, harassment complaints, leave approval, payroll compliance, offboarding, training, code of conduct, remote work, data privacy, overtime, workplace safety, hiring bias, promotion fairness, benefits, engagement, conflicts of interest, discipline |

## Program Structure

| File | Description |
|------|-------------|
| `generate_dataset.py` | main program: three-model architecture, multi-domain support, config-driven |
| `generate_config.py` | step 1: generate config + document-driven question templates from the policy document |
| `evaluate_dataset.py` | dataset quality evaluation (schema validation, type distribution, gold/teacher agreement) |
| `llm_client.py` | OpenAI-compatible LLM client (chat/chat_json, retries and JSON extraction) |
| `template_store.py` | per-domain template load/save (`templates/{domain}.json`, deduplicated merge by key) |
| `gold_rules.py` | deterministic gold derivation from state-embedded facts (identity/map/count_gte/gte/lte/eq/in) |
| `teacher.py` | teacher probability validation, normalization and ensemble averaging |
| `config_loader.py` | three-model endpoint config loading (config.json + env var overrides) |
| `templates/{domain}.json` | per-domain question templates (**the single source of truth**; e.g. 23 for compliance) |
| `question_templates.json` | pre-migration backup (old format), no longer read by the program |
| `scripts/split_templates.py` | splits the old `question_templates.json` into per-domain `templates/` files |
| `tests/` | unit tests (gold_rules, template_store, teacher, schema validation, etc.) |
| `config.json` | three independent LLM endpoint configs |

## Usage Example

```bash
# 1. Generate config + document-driven templates
python generate_config.py --pdf <path-to-policy-doc>.pdf --domain compliance

# 2. Generate the dataset
python generate_dataset.py --pdf <path-to-policy-doc>.pdf --target 10 --domain compliance --output dataset_10.json

# 3. Evaluate the dataset
python evaluate_dataset.py --input dataset_10.json
```

## Output Example

```json
{
  "state_id": "synth-xxx",
  "generator": "v1-compliance",
  "domain": "compliance",
  "n_questions": 3,
  "state": "{\"description\": \"...\", \"client_info\": {...}, ...}",
  "state_is_json": true,
  "questions": "{\"account_compliant\": {\"type\": \"choice\", \"instructions\": \"Determine whether this account opening process complies with company compliance policy.\", \"criteria\": {\"compliant\": \"...\", \"non-compliant\": \"...\", \"needs-review\": \"...\"}}}",
  "gold": "{\"account_compliant\": \"compliant\"}",
  "teacher": "{\"account_compliant\": {\"probabilities\": {\"compliant\": 0.95, \"non-compliant\": 0.03, \"needs-review\": 0.02}}}",
  "teacher_model": "Qwen3.6-35B-A3B"
}
```

Note: `questions` / `gold` / `teacher` are **JSON strings** (as shown above); consumers must parse them with `json.loads`.

### How to Add a New Domain

To add a new domain (e.g. "finance"), modify the following:

1. **`templates/finance.json`** — create the Jev three-primitive question template list for the domain (every template must carry a `facts` spec, otherwise it does not participate in question selection):
   ```json
   [
     {
       "key": "loan_approval",
       "type": "choice",
       "instructions": "Determine whether the loan application meets approval criteria.",
       "criteria": {"approved": "...", "rejected": "...", "needs-review": "..."},
       "facts": {"field": "loan.status", "rule": {"op": "identity"}},
       "policy_keywords": ["Loan Approval", "Credit"]
     }
   ]
   ```

2. **`generate_config.py`** — add domain-specific information in the following places:
   - `domain_keywords` dict: keyword list for the domain
   - `domain_field_names` dict: field names of the state dictionary
   - `default_decision_points` dict: default decision points
   - `domain_names` dict: display name and document name of the domain
   - `domain_keywords_for_extraction` dict: keywords used for section extraction

3. **Generate config and test**:
   ```bash
   python generate_config.py --domain finance --pdf <path-to-finance-manual>.pdf
   python generate_dataset.py --target 10 --domain finance --output dataset_finance_10.json
   ```

# Notes

1. Generating large sample counts takes a long time; run in batches (5–10 per batch)
2. Make sure the LLM server is running and the model is available
3. Generated data is for training purposes only; review it manually before production use
4. The program extracts the decision-relevant sections of the policy document by default (~30K characters)
5. Each sample contains 3–4 randomly selected questions covering different topics
6. The teacher model uses Qwen3.6-35B-A3B with temperature=0.3 to obtain probabilistic predictions
7. gold is derived deterministically from state-embedded facts by rules; a question that cannot be derived is skipped, and if no question in a sample is derivable the sample is dropped and counted as a degradation
8. `--resume` continues from an existing JSONL up to target (already-completed samples are not regenerated)
9. After 3 consecutive state-generation failures the program aborts; completed work is kept in the JSONL and can be resumed with `--resume`
10. Samples that produce no questions at all are counted as degradations (reported on the completion line) and are not written to the dataset
11. Templates without a facts spec do not participate in question selection (only templates with facts enter the sampling pool)

## Environment Requirements

- Python 3.8+
- requests
- pdfminer.six (for PDF text extraction)

```bash
pip install requests pdfminer.six
```
