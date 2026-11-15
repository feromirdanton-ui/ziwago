"""Регрессия: режим приёмки доходит до гейта реестра ранкера; прогресс пишется атомарно."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_ziwago_v19 as runner  # noqa: E402


def _fake_assistant(ledger_scope: str | None):
    metadata = {} if ledger_scope is None else {"identity_ledger_scope": ledger_scope}
    genesis = SimpleNamespace(
        authority=SimpleNamespace(records=[]),
        document_ranker=object(),
        document_ranker_metadata=metadata,
    )
    # Намеренно без evaluation_context: если гейт НЕ сработал, вызов упадёт
    # с AttributeError, а не с ожидаемым ValueError.
    return SimpleNamespace(genesis=genesis, organism=SimpleNamespace(language_memory_records=[]))


def test_open_book_incomplete_ledger_is_rejected_in_frozen_evaluation():
    assistant = _fake_assistant("partial")
    with pytest.raises(ValueError, match="complete selected-identity ledger"):
        runner.evaluate_frozen_acceptance(assistant, [], [], "open_book")


def test_closed_book_does_not_require_ranker_ledger():
    assistant = _fake_assistant("partial")
    assert runner.acceptance_ranker_metadata(assistant.genesis, "closed_book") == {
        "identity_ledger_scope": "partial"}


def test_open_book_complete_ledger_passes_the_gate():
    scope = "all selected question/context identities, including skipped and filtered groups"
    assistant = _fake_assistant(scope)
    assert runner.acceptance_ranker_metadata(assistant.genesis, "open_book")[
        "identity_ledger_scope"] == scope


def test_write_json_atomic_replaces_file_and_leaves_no_temp(tmp_path):
    target = tmp_path / "progress.json"
    runner._write_json_atomic(target, {"stage": "one", "text": "кириллица"})
    runner._write_json_atomic(target, {"stage": "two"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"stage": "two"}
    assert [p.name for p in tmp_path.iterdir()] == ["progress.json"]
