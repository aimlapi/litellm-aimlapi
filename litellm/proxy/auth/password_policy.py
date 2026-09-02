"""Password-strength policy enforcement for locally-managed proxy users.

Applied at every path that persists a new or changed password for a DB-backed
user (``/user/update``, ``/user/bulk_update``, and the invitation onboarding
claim flow), so the strength bar is configured in one place instead of
per-endpoint.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from litellm.proxy._types import ProxyErrorTypes, ProxyException

DEFAULT_MIN_LENGTH: Final = 12

_UPPERCASE_RE: Final = re.compile(r"[A-Z]")
_LOWERCASE_RE: Final = re.compile(r"[a-z]")
_DIGIT_RE: Final = re.compile(r"[0-9]")
_SPECIAL_RE: Final = re.compile(r"[^A-Za-z0-9]")


@dataclass(frozen=True, slots=True)
class PasswordPolicy:
    min_length: int
    require_uppercase: bool
    require_lowercase: bool
    require_numbers: bool
    require_special_characters: bool


def get_password_policy(general_settings: Mapping[str, object]) -> PasswordPolicy:
    min_length_setting: Final = general_settings.get("password_policy_min_length")
    return PasswordPolicy(
        min_length=(int(min_length_setting) if isinstance(min_length_setting, (int, float)) else DEFAULT_MIN_LENGTH),
        require_uppercase=general_settings.get("password_policy_require_uppercase", True) is not False,
        require_lowercase=general_settings.get("password_policy_require_lowercase", True) is not False,
        require_numbers=general_settings.get("password_policy_require_numbers", True) is not False,
        require_special_characters=(
            general_settings.get("password_policy_require_special_characters", True) is not False
        ),
    )


def _policy_violations(password: str, policy: PasswordPolicy) -> tuple[str, ...]:
    checks: Final = (
        (len(password) < policy.min_length, f"be at least {policy.min_length} characters long"),
        (policy.require_uppercase and not _UPPERCASE_RE.search(password), "include an uppercase letter"),
        (policy.require_lowercase and not _LOWERCASE_RE.search(password), "include a lowercase letter"),
        (policy.require_numbers and not _DIGIT_RE.search(password), "include a number"),
        (policy.require_special_characters and not _SPECIAL_RE.search(password), "include a special character"),
    )
    return tuple(message for failed, message in checks if failed)


def validate_password_policy(password: str, general_settings: Mapping[str, object]) -> None:
    """Raise ``ProxyException`` (400) if ``password`` fails the configured policy."""
    policy: Final = get_password_policy(general_settings)
    violations: Final = _policy_violations(password, policy)
    if not violations:
        return
    raise ProxyException(
        message="Password does not meet the required policy: must " + ", ".join(violations) + ".",
        type=ProxyErrorTypes.validation_error,
        param="password",
        code=400,
    )
