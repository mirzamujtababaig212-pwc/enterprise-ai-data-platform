"""
Utilities for sanitizing sensitive information before logging.

This module provides reusable helper functions for masking sensitive
values before they are written to application logs.
"""

from typing import Any
import re

# Constant used to replace sensitive values.
REDACTED = "********"

# Headers whose values should never appear in logs.
SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-openai-api-key",
    "openai-api-key",
    "anthropic-api-key",
    "gemini-api-key",
    "azure-openai-key",
    "cookie",
    "set-cookie",
}

# JSON body fields whose values should never appear in logs.
SENSITIVE_BODY_FIELDS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "secret",
    "client_secret",
    "client_id",
    "token",
    "bearer_token",
}

SENSITIVE_TEXT_PATTERNS = [
    # OpenAI-style API keys
    re.compile(
        r"sk-[A-Za-z0-9_-]+",
        re.IGNORECASE,
    ),
    # Bearer tokens
    re.compile(
        r"Bearer\s+[A-Za-z0-9._-]+",
        re.IGNORECASE,
    ),
    # Generic API key assignment
    re.compile(
        r"(api[_-]?key\s*[:=]\s*)\S+",
        re.IGNORECASE,
    ),
    # Authorization header values
    re.compile(
        r"(authorization\s*[:=]\s*)\S+",
        re.IGNORECASE,
    ),
]


def sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Return a sanitized copy of HTTP headers for safe logging.

    Sensitive header values are replaced with the REDACTED constant.
    Header names are compared in a case-insensitive manner while
    preserving their original capitalization.

    A new dictionary is returned and the original dictionary is
    never modified.
    """

    # Create a shallow copy so the caller's dictionary is not modified.
    sanitized_headers = headers.copy()

    # Iterate over the copied dictionary.
    for header_name, header_value in sanitized_headers.items():

        # Normalize the header name for case-insensitive comparison.
        normalized_header = header_name.lower()

        # Mask sensitive header values.
        if normalized_header in SENSITIVE_HEADERS:
            sanitized_headers[header_name] = REDACTED

    return sanitized_headers


def sanitize_body(body: Any) -> Any:
    """
    Recursively sanitize a JSON-compatible request body for safe logging.

    This function traverses dictionaries and lists of arbitrary depth,
    replacing values of sensitive fields with REDACTED while preserving
    the original structure.

    The original object is never modified.
    """

    # Dictionary
    if isinstance(body, dict):

        sanitized = {}

        for key, value in body.items():

            normalized_key = key.lower()

            if normalized_key in SENSITIVE_BODY_FIELDS:
                sanitized[key] = REDACTED
            else:
                sanitized[key] = sanitize_body(value)

        return sanitized

    # List
    if isinstance(body, list):

        return [sanitize_body(item) for item in body]

    # Strings
    if isinstance(body, str):
        return sanitize_text(body)

    # Primitive value
    return body


def sanitize_text(text: str) -> str:
    """
    Return a sanitized copy of free-form text for safe logging.

    Secrets embedded within arbitrary text are replaced with the
    REDACTED constant using configurable regular-expression
    patterns.

    A new string is returned and the original string remains
    unchanged.
    """

    sanitized_text = text

    for pattern in SENSITIVE_TEXT_PATTERNS:
        sanitized_text = pattern.sub(
            REDACTED,
            sanitized_text,
        )

    return sanitized_text
