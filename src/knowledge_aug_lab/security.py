"""Security controls that must run before retrieval and generation."""

from __future__ import annotations

from collections.abc import Iterable

from knowledge_aug_lab.models import Document


def filter_authorized_documents(
    documents: Iterable[Document],
    scopes: set[str],
    trusted_only: bool = True,
) -> list[Document]:
    """Apply source trust and ACL metadata before indexing or ranking."""

    if not isinstance(scopes, (set, frozenset)) or any(
        not isinstance(scope, str) or not scope.strip() for scope in scopes
    ):
        raise ValueError("request scopes must be a set of nonempty strings")
    if not isinstance(trusted_only, bool):
        raise ValueError("trusted_only must be a bool")

    allowed: list[Document] = []
    for document in documents:
        if trusted_only and document.metadata.get("trusted") is not True:
            continue
        raw_scopes = document.metadata.get("scopes")
        if not isinstance(raw_scopes, (list, tuple, set, frozenset)):
            continue
        if not raw_scopes or any(not isinstance(scope, str) or not scope.strip() for scope in raw_scopes):
            continue
        document_scopes = set(raw_scopes)
        if not document_scopes or document_scopes.isdisjoint(scopes):
            continue
        allowed.append(document)
    return allowed
