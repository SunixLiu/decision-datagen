from generate_dataset import build_policy_digest, split_sections

POLICY = """
Account Opening
All accounts require KYC.

Anti Money Laundering
Report suspicious activity.

Privacy Policy
Protect client data.
"""


def test_split_sections():
    sections = split_sections(POLICY)
    titles = [t for t, _ in sections]
    assert "Account Opening" in titles
    assert "Anti Money Laundering" in titles


def test_digest_matches_keywords():
    templates = {
        "q1": {"policy_keywords": ["Money Laundering"]},
        "q2": {"policy_keywords": ["NonexistentKeyword"]},
    }
    digest = build_policy_digest(POLICY, templates, max_chars=500)
    assert "Anti Money Laundering" in digest["q1"]
    assert "Account Opening" not in digest["q1"]
    # 无匹配时回退全文
    assert "Account Opening" in digest["q2"]
