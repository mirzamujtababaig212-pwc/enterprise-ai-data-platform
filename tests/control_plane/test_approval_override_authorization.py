from __future__ import annotations

import pytest

from app.control_plane.approvals.authorization import (
    ConfiguredApprovalOverrideAuthorizer,
)


def test_configured_principal_is_authorized() -> None:
    authorizer = ConfiguredApprovalOverrideAuthorizer(
        frozenset({"api_key:operator-1"}),
    )

    assert authorizer.is_authorized("api_key:operator-1") is True


def test_unconfigured_principal_is_not_authorized() -> None:
    authorizer = ConfiguredApprovalOverrideAuthorizer(
        frozenset({"api_key:operator-1"}),
    )

    assert authorizer.is_authorized("api_key:other") is False


def test_empty_configuration_denies_all_principals() -> None:
    authorizer = ConfiguredApprovalOverrideAuthorizer(frozenset())

    assert authorizer.is_authorized("api_key:operator-1") is False


def test_empty_principal_is_rejected() -> None:
    authorizer = ConfiguredApprovalOverrideAuthorizer(
        frozenset({"api_key:operator-1"}),
    )

    with pytest.raises(ValueError, match="principal must not be empty"):
        authorizer.is_authorized("   ")
