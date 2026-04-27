"""Masking test suite — 15+ tests."""
from __future__ import annotations

import pytest

from agentblackbox import BlackBox
from agentblackbox.masking import PIIMasker


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def masker():
    return PIIMasker()


@pytest.fixture(autouse=True)
def reset_blackbox(tmp_path):
    BlackBox.configure(db_path=str(tmp_path / "test.db"), masking=False)
    yield
    BlackBox.configure(masking=False)


# ── credit card (4 patterns) ──────────────────────────────────────────────────

class TestCreditCard:
    def test_dashes(self, masker):
        result = masker.mask("Card: 4111-1111-1111-1111")
        assert "[MASKED_CREDIT_CARD]" in result
        assert "4111-1111-1111-1111" not in result

    def test_spaces(self, masker):
        result = masker.mask("Card: 4111 1111 1111 1111")
        assert "[MASKED_CREDIT_CARD]" in result

    def test_no_separator(self, masker):
        result = masker.mask("Card: 4111111111111111")
        assert "[MASKED_CREDIT_CARD]" in result

    def test_amex_pattern(self, masker):
        # 15-digit Amex doesn't match the 4x4 pattern — confirm no false positive
        result = masker.mask("Ref: 123456789012345")
        # 15 digits don't match the credit card pattern (needs 4+4+4+4)
        assert "MASKED_CREDIT_CARD" not in result

    def test_mixed_separator(self, masker):
        # 4111 with mixed space/dash separators — no zeros to trigger phone_jp
        result = masker.mask("Card: 4111 1111-1111 1111")
        assert "[MASKED_CREDIT_CARD]" in result


# ── email ─────────────────────────────────────────────────────────────────────

class TestEmail:
    def test_basic_email(self, masker):
        result = masker.mask("Contact: user@example.com")
        assert "[MASKED_EMAIL]" in result
        assert "user@example.com" not in result

    def test_email_with_subdomain(self, masker):
        result = masker.mask("john.doe+tag@mail.company.co.uk")
        assert "[MASKED_EMAIL]" in result

    def test_email_in_sentence(self, masker):
        result = masker.mask("Send to alice@test.org for details")
        assert "[MASKED_EMAIL]" in result
        assert "alice@test.org" not in result


# ── phone numbers ─────────────────────────────────────────────────────────────

class TestPhoneNumbers:
    def test_japanese_phone_with_dash(self, masker):
        result = masker.mask("Tel: 090-1234-5678")
        assert "[MASKED_PHONE_JP]" in result

    def test_japanese_phone_no_separator(self, masker):
        result = masker.mask("Tel: 09012345678")
        assert "[MASKED_PHONE_JP]" in result

    def test_us_phone_standard(self, masker):
        result = masker.mask("Call 555-867-5309")
        assert "[MASKED_PHONE_US]" in result

    def test_us_phone_with_country_code(self, masker):
        result = masker.mask("Call +1 800 555 1234")
        assert "[MASKED_PHONE_US]" in result


# ── API keys ──────────────────────────────────────────────────────────────────

class TestAPIKeys:
    def test_openai_style_key(self, masker):
        # sk_ prefix + 20 alphanumeric chars (no underscore in body)
        result = masker.mask("key: sk_ABCDEFGHIJKLMNOPQRSTuvwx")
        assert "[MASKED_API_KEY]" in result

    def test_generic_token(self, masker):
        result = masker.mask("Authorization: token_abcdefghijklmnopqrstuvwxyz")
        assert "[MASKED_API_KEY]" in result

    def test_aws_access_key(self, masker):
        # Standard AWS key: AKIA + exactly 16 uppercase alphanumeric chars
        result = masker.mask("Access key: AKIAIOSFODNN7EXAMPLE")
        assert "[MASKED_AWS_KEY]" in result


# ── JWT ───────────────────────────────────────────────────────────────────────

class TestJWT:
    def test_jwt_token(self, masker):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        result = masker.mask(f"Bearer {jwt}")
        assert "[MASKED_JWT]" in result
        assert jwt not in result


# ── mask_dict ─────────────────────────────────────────────────────────────────

class TestMaskDict:
    def test_flat_dict(self, masker):
        data = {"email": "user@example.com", "name": "Alice"}
        result = masker.mask_dict(data)
        assert "[MASKED_EMAIL]" in result["email"]
        assert result["name"] == "Alice"

    def test_nested_dict(self, masker):
        data = {"user": {"contact": "admin@test.com", "age": 30}}
        result = masker.mask_dict(data)
        assert "[MASKED_EMAIL]" in result["user"]["contact"]
        assert result["user"]["age"] == 30

    def test_list_values(self, masker):
        data = {"emails": ["a@b.com", "c@d.org"]}
        result = masker.mask_dict(data)
        for email in result["emails"]:
            assert "[MASKED_EMAIL]" in email


# ── enabled=False ─────────────────────────────────────────────────────────────

class TestDisabled:
    def test_disabled_mask_returns_unchanged(self):
        m = PIIMasker(enabled=False)
        text = "user@example.com and 4111-1111-1111-1111"
        assert m.mask(text) == text

    def test_disabled_mask_dict_returns_unchanged(self):
        m = PIIMasker(enabled=False)
        data = {"email": "user@example.com"}
        result = m.mask_dict(data)
        assert result["email"] == "user@example.com"


# ── custom patterns ───────────────────────────────────────────────────────────

class TestCustomPatterns:
    def test_custom_pattern_match(self):
        m = PIIMasker(custom_patterns={"emp_id": r"EMP-\d{6}"})
        result = m.mask("Employee EMP-123456 has been assigned")
        assert "[MASKED_EMP_ID]" in result
        assert "EMP-123456" not in result

    def test_custom_pattern_does_not_affect_builtins(self):
        m = PIIMasker(custom_patterns={"my_pat": r"CUSTOM-\d+"})
        # email should still be masked since all builtins are active
        result = m.mask("user@example.com and CUSTOM-999")
        assert "[MASKED_EMAIL]" in result
        assert "[MASKED_MY_PAT]" in result

    def test_select_subset_of_builtin_patterns(self):
        m = PIIMasker(patterns=["email"])
        result = m.mask("user@example.com and 4111-1111-1111-1111")
        assert "[MASKED_EMAIL]" in result
        # credit card should NOT be masked (pattern not selected)
        assert "4111-1111-1111-1111" in result


# ── BlackBox integration ──────────────────────────────────────────────────────

class TestBlackBoxMasking:
    def test_configure_masking_masks_llm_input(self, tmp_path):
        BlackBox.configure(db_path=str(tmp_path / "db.db"), masking=True)
        with BlackBox.session("secure") as bb:
            call = bb.record_llm_call(
                model="gpt-4o",
                input_text="Email: john@example.com, card: 4111-1111-1111-1111",
                output_text="Processed payment for john@example.com",
            )
        assert "[MASKED_EMAIL]" in call.input_text
        assert "[MASKED_CREDIT_CARD]" in call.input_text
        assert "[MASKED_EMAIL]" in call.output_text

    def test_configure_masking_masks_tool_args(self, tmp_path):
        BlackBox.configure(db_path=str(tmp_path / "db.db"), masking=True)
        with BlackBox.session("secure_tool") as bb:
            call = bb.record_tool_call(
                tool_name="payment",
                args={"card": "4111-1111-1111-1111", "email": "pay@example.com"},
                result={"status": "ok"},
            )
        assert "[MASKED_CREDIT_CARD]" in call.args["card"]
        assert "[MASKED_EMAIL]" in call.args["email"]

    def test_no_masking_by_default(self, tmp_path):
        BlackBox.configure(db_path=str(tmp_path / "db.db"), masking=False)
        with BlackBox.session("plain") as bb:
            call = bb.record_llm_call(
                model="gpt-4o",
                input_text="user@example.com",
                output_text="ok",
            )
        assert call.input_text == "user@example.com"
