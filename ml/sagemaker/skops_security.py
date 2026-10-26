from __future__ import annotations

from pathlib import Path

import skops.io as sio

# This type is required by supported scikit-learn tree-based estimators.
# Any additional SKOPS type must be reviewed and explicitly approved here.
ALLOWED_UNTRUSTED_SKOPS_TYPES = frozenset(
    {
        "sklearn.tree._tree.Tree",
    }
)


def get_approved_untrusted_types(
    model_path: str | Path,
    *,
    context: str,
) -> list[str]:
    """Return explicitly approved SKOPS types or reject the artifact."""
    reported_types = sio.get_untrusted_types(file=str(model_path))
    unapproved_types = sorted(set(reported_types) - ALLOWED_UNTRUSTED_SKOPS_TYPES)

    if unapproved_types:
        raise ValueError(f"{context}: " + ", ".join(unapproved_types))

    # Pass only types actually reported by SKOPS and explicitly allowlisted.
    return sorted(set(reported_types))
