"""
Tests for the configurable password-strength policy in
`litellm.proxy.auth.password_policy`, enforced on every path that persists a
new or changed password for a locally-managed user.
"""

import pytest

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.password_policy import (
    DEFAULT_MIN_LENGTH,
    PasswordPolicy,
    get_password_policy,
    validate_password_policy,
)

STRONG_PASSWORD = "Str0ng!Passw0rd"


def test_get_password_policy_defaults_to_pif_baseline():
    policy = get_password_policy({})
    assert policy == PasswordPolicy(
        min_length=DEFAULT_MIN_LENGTH,
        require_uppercase=True,
        require_lowercase=True,
        require_numbers=True,
        require_special_characters=True,
    )


def test_get_password_policy_reads_overrides_from_general_settings():
    policy = get_password_policy(
        {
            "password_policy_min_length": 20,
            "password_policy_require_uppercase": False,
            "password_policy_require_lowercase": False,
            "password_policy_require_numbers": False,
            "password_policy_require_special_characters": False,
        }
    )
    assert policy == PasswordPolicy(
        min_length=20,
        require_uppercase=False,
        require_lowercase=False,
        require_numbers=False,
        require_special_characters=False,
    )


def test_validate_password_policy_accepts_strong_password():
    # Must not raise.
    validate_password_policy(STRONG_PASSWORD, {})


@pytest.mark.parametrize(
    "password,expected_fragment",
    [
        ("Sh0rt!Pw", "12 characters"),
        ("weakpassword123!", "uppercase"),
        ("WEAKPASSWORD123!", "lowercase"),
        ("WeakPassword!!!!", "number"),
        ("WeakPassword12345", "special character"),
    ],
)
def test_validate_password_policy_rejects_each_missing_class(password, expected_fragment):
    with pytest.raises(ProxyException) as exc_info:
        validate_password_policy(password, {})
    assert exc_info.value.code == "400"
    assert exc_info.value.type == ProxyErrorTypes.validation_error
    assert exc_info.value.param == "password"
    assert expected_fragment in exc_info.value.message


def test_validate_password_policy_reports_every_violation_at_once():
    with pytest.raises(ProxyException) as exc_info:
        validate_password_policy("weak", {})
    assert "12 characters" in exc_info.value.message
    assert "uppercase" in exc_info.value.message
    assert "number" in exc_info.value.message
    assert "special character" in exc_info.value.message


def test_validate_password_policy_honors_relaxed_config():
    general_settings = {
        "password_policy_min_length": 6,
        "password_policy_require_special_characters": False,
    }
    # 6 chars, has upper/lower/number, no special char: fails default policy,
    # passes the relaxed one above.
    validate_password_policy("Abcd12", general_settings)
    with pytest.raises(ProxyException):
        validate_password_policy("Abcd12", {})


def test_validate_password_policy_honors_stricter_min_length():
    general_settings = {"password_policy_min_length": 20}
    with pytest.raises(ProxyException) as exc_info:
        validate_password_policy(STRONG_PASSWORD, general_settings)
    assert "20 characters" in exc_info.value.message
