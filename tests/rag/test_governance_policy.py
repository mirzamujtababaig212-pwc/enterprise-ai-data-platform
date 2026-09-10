from rag.governance import GovernancePolicy


def test_empty_policy_produces_empty_metadata_filter():
    policy = GovernancePolicy()

    assert policy.to_metadata_filter() == {}


def test_policy_preserves_required_metadata():
    policy = GovernancePolicy(
        required_metadata={
            "classification": "internal",
            "allowed_use": "enterprise_ai",
            "domain": "mobility",
        }
    )

    assert policy.to_metadata_filter() == {
        "classification": "internal",
        "allowed_use": "enterprise_ai",
        "domain": "mobility",
    }


def test_policy_returns_copy_of_required_metadata():
    policy = GovernancePolicy(
        required_metadata={
            "classification": "internal",
        }
    )

    metadata_filter = policy.to_metadata_filter()
    metadata_filter["classification"] = "restricted"

    assert policy.to_metadata_filter() == {
        "classification": "internal",
    }


def test_policy_rejects_empty_metadata_key():
    try:
        GovernancePolicy(
            required_metadata={
                "": "internal",
            }
        )
    except ValueError as exc:
        assert str(exc) == "Governance metadata keys must be non-empty strings."
    else:
        raise AssertionError("Expected ValueError")


def test_policy_rejects_whitespace_metadata_key():
    try:
        GovernancePolicy(
            required_metadata={
                "   ": "internal",
            }
        )
    except ValueError as exc:
        assert str(exc) == "Governance metadata keys must be non-empty strings."
    else:
        raise AssertionError("Expected ValueError")
