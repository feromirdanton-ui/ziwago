#!/usr/bin/env python3
"""Train from explicit local splits, evaluate frozen candidates, never overwrite a master."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from ziwago.evaluation.gates import (
    _normal_prompt, evaluate_acceptance_suite, evaluate_grammar_and_artifacts, validate_acceptance_documents,
)
from train_ziwago_v14_pure_synaptic_dialogue import TokenizedDialogPair
from ziwago.organism_core import ROOT
from ziwago.organism.verified import accept_candidate, file_sha256, stable_hash
from ziwago_assistant import ZiwagoBiologicalSNNAssistant


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Записать JSON атомарно: temp в том же каталоге + fsync + os.replace.

    Длинные прогоны убиваются рантаймом (~40 мин); обычный `write_text` мог
    оставить обрезанный файл прогресса, и по нему нельзя было понять стадию.
    """
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_splits(dataset: Path, acceptance: Path, acceptance_mode: str = "closed_book", *,
                dataset_snapshot: bytes | None = None, acceptance_snapshot: bytes | None = None) -> tuple[list[dict], list[dict]]:
    manifest_path = dataset.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("eligible_for_future_training") is False:
            raise ValueError("The dataset manifest marks this curriculum as invalid for training.")
    dataset_bytes = dataset.read_bytes() if dataset_snapshot is None else dataset_snapshot
    acceptance_bytes = acceptance.read_bytes() if acceptance_snapshot is None else acceptance_snapshot
    rows = [(n, json.loads(line)) for n, line in enumerate(dataset_bytes.decode("utf-8").splitlines(), 1) if line.strip()]
    train = [item for _, item in rows]
    cases = json.loads(acceptance_bytes)
    if not isinstance(cases, list) or not train or not cases:
        raise ValueError("Nonempty training JSONL and acceptance JSON array required.")
    for line_number, item in rows:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k].strip() for k in ("prompt", "response", "family_id", "source")):
            raise ValueError("Training records require prompt, response, source and semantic family_id.")
        if item.get("split", "train") != "train":
            raise ValueError("Non-training record in training file.")
        item["_source_line"] = line_number
    for case in cases:
        if (not isinstance(case, dict)
                or not all(isinstance(case.get(k), str) and case[k].strip() for k in ("prompt", "family_id"))
                or not isinstance(case.get("expected_concepts"), list) or not case["expected_concepts"]
                or not all(isinstance(c, str) and c.strip() for c in case["expected_concepts"])):
            raise ValueError("Acceptance cases require prompt, expected_concepts and independent family_id.")
    train_families = {item["family_id"] for item in train}
    acceptance_families = {case["family_id"] for case in cases}
    if train_families & acceptance_families:
        raise ValueError("Train/acceptance semantic families overlap.")
    train_prompts = {" ".join(item["prompt"].lower().split()) for item in train}
    if any(" ".join(c["prompt"].lower().split()) in train_prompts for c in cases):
        raise ValueError("Acceptance prompt leaked into training.")
    if len(cases) < 2:
        raise ValueError("At least two acceptance cases required before training.")
    document_errors = validate_acceptance_documents(
        cases, acceptance_mode,
        trained_documents=[item["document"] for item in train if isinstance(item.get("document"), str)],
        trained_document_hashes=[item["provenance"][key] for item in train
                                 for key in ("context_sha256", "normalized_context_sha256")
                                 if item.get("provenance", {}).get(key)])
    if document_errors:
        raise ValueError("; ".join(document_errors))
    return train, cases


def acceptance_ranker_metadata(runtime, acceptance_mode: str = "closed_book") -> dict:
    """Реестр идентичностей ранкера документов.

    Гейт относится к ОТКРЫТОМУ режиму приёмки (`open_book`): там ранкер
    ранжирует предложения документа, и неполный реестр identичностей означал
    бы, что часть отобранных вопросов ранкер не видел. В `closed_book` ранкер
    документов вообще не используется, и требование полного реестра там не
    имеет смысла.

    Раньше проверка была безусловной — и заблокировала обучение кандидата,
    унаследовавшего ранкер со старым (неполным) реестром: получался
    порочный круг, где строгость приёмок отключает саму приёмку. Фактическое
    состояние реестра в любом случае попадает в отчёт.
    """
    metadata = runtime.document_ranker_metadata or {}
    if (acceptance_mode == "open_book" and runtime.document_ranker is not None
            and metadata.get("identity_ledger_scope")
            != "all selected question/context identities, including skipped and filtered groups"):
        raise ValueError("Acceptance requires a complete selected-identity ledger for the document ranker.")
    return metadata


def evaluate_frozen_acceptance(assistant, train: list[dict], cases: list[dict], mode: str) -> dict:
    def generate(prompt: str, *, document: str | None = None):
        if mode == "open_book":
            return assistant.chat_with_document(prompt, document)
        return assistant.chat(prompt)

    known_records = [item for item in train + assistant.genesis.authority.records + assistant.organism.language_memory_records
                     if item.get("split", "train") == "train"]
    # Режим обязан попасть в гейт: без него open_book-проверка реестра ранкера
    # здесь никогда не срабатывала (дефолт — closed_book).
    ranker_metadata = acceptance_ranker_metadata(assistant.genesis, mode)
    ranker_families = {family for key in ("fit_family_ids", "internal_dev_family_ids", "calibration_family_ids")
                       for family in ranker_metadata.get(key, [])}
    known_families = {item.get("family_id") for item in known_records if item.get("family_id")} | ranker_families
    if known_families & {case["family_id"] for case in cases}:
        raise ValueError("Candidate training/acceptance families overlap.")
    with assistant.evaluation_context():
        report = evaluate_acceptance_suite(
            generate, cases, trained_prompts=[item["prompt"] for item in known_records], mode=mode,
            trained_prompt_hashes=ranker_metadata.get("exposed_prompt_hashes", []),
            trained_documents=[item["document"] for item in known_records if isinstance(item.get("document"), str)],
            trained_document_hashes=[item["provenance"][key] for item in known_records
                                     for key in ("context_sha256", "normalized_context_sha256")
                                     if item.get("provenance", {}).get(key)] + ranker_metadata.get("exposed_document_hashes", []))
    ranker_hash = ranker_metadata.get("ranker_checkpoint_sha256")
    report["document_ranker_sha256"] = ranker_hash
    if ranker_hash is not None:
        report["split_hash"] = stable_hash({"acceptance_split_hash": report["split_hash"], "document_ranker_sha256": ranker_hash})
    return report


def run_evaluation(dataset: Path, acceptance: Path, checkpoint: Path,
                   output_dir: Path = ROOT / "Produce", seed: int = 42,
                   workers: int = 4, cpu_threads: int = 8, acceptance_mode: str = "closed_book",
                   document_ranker_checkpoint: Path | None = None, runtime_version: int = 19) -> dict:
    if runtime_version not in (19, 20):
        raise ValueError("runtime_version must be 19 or 20")
    output_dir, checkpoint = Path(output_dir), Path(checkpoint)
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    dataset_bytes, acceptance_bytes = Path(dataset).read_bytes(), Path(acceptance).read_bytes()
    train, cases = load_splits(Path(dataset), Path(acceptance), acceptance_mode,
                              dataset_snapshot=dataset_bytes, acceptance_snapshot=acceptance_bytes)
    source_hashes = {"training": hashlib.sha256(dataset_bytes).hexdigest(),
                     "acceptance": hashlib.sha256(acceptance_bytes).hexdigest()}
    checkpoint_hash = file_sha256(checkpoint)
    assistant = ZiwagoBiologicalSNNAssistant(root=ROOT, checkpoint_path=checkpoint, seed=seed, workers=workers,
                                            cpu_threads=cpu_threads, expected_checkpoint_sha256=checkpoint_hash,
                                            runtime_version=runtime_version)
    ranker_info = assistant.genesis.load_document_ranker(document_ranker_checkpoint) if document_ranker_checkpoint is not None else None
    if (file_sha256(checkpoint) != checkpoint_hash
            or file_sha256(dataset) != source_hashes["training"]
            or file_sha256(acceptance) != source_hashes["acceptance"]):
        raise RuntimeError("An input changed while constructing the evaluation runtime.")
    report = evaluate_frozen_acceptance(assistant, train, cases, acceptance_mode)
    if (file_sha256(checkpoint) != checkpoint_hash
            or file_sha256(dataset) != source_hashes["training"]
            or file_sha256(acceptance) != source_hashes["acceptance"]):
        raise RuntimeError("An evaluated input changed during read-only evaluation.")
    if ranker_info is not None and file_sha256(document_ranker_checkpoint) != ranker_info["sha256"]:
        raise RuntimeError("Document ranker checkpoint changed during evaluation.")
    report.update({"checkpoint_sha256": checkpoint_hash, "source_hashes": source_hashes,
                   "train_family_hash": stable_hash(sorted({item["family_id"] for item in train})),
                   "full_corpus_training_certified": False})
    result = {"version": f"{runtime_version}.0", "checkpoint": str(checkpoint.resolve()), "acceptance": report,
              "training_performed": False, "checkpoint_unchanged": True, "legacy_master_unchanged": True}
    if runtime_version == 20:
        result["organism"] = assistant.organism.life.telemetry(assistant.organism)
    report_path = output_dir / f"v{runtime_version}_{acceptance_mode}_evaluation_{time.time_ns()}.json"
    result["report_path"] = str(report_path)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def measure_language_diagnostics(assistant, records: list[dict], split: str) -> dict:
    core = assistant.organism
    rows = []
    for item in records[:4]:
        reference = item.get("response", item.get("reference"))
        if reference is None:
            continue
        likelihood = core.teacher_forced_likelihood(item["prompt"], reference)
        row = {"prompt": item["prompt"], "family_id": item["family_id"], "likelihood": likelihood}
        if split == "train":
            with core.evaluation_context():
                q_vec = core.compute_stimulus_vec64(core.encode_text(item["prompt"]))
                core._prepare_language_sequence(q_vec)
                ids, _, _ = core._synthesize_single_beam_from_weights_only(
                    q_vec, q_vec, (0, 0), max_tokens=core.dendritic_plateau_readout.max_pos, use_ngrams=False)
                response = core.decode_tokens(ids).strip()
            row.update({"raw_weight_response": response,
                        "exact_reference_match": " ".join(response.split()) == " ".join(reference.split()),
                        "source_supported": bool(assistant.genesis.authority.ground(item["prompt"], response)["grounded"])})
        rows.append(row)
    measured = [row["likelihood"] for row in rows if row["likelihood"]["available"]]
    count = sum(row["token_count"] for row in measured)
    return {"split": split, "rows": rows, "available_records": len(measured),
            "mean_nll_nats": sum(row["nll_sum"] for row in measured) / count if count else None,
            "token_error_rate": sum(row["token_errors"] for row in measured) / count if count else None,
            "independent_acceptance": False}


def run_training(dataset: Path, acceptance: Path, output_dir: Path = ROOT / "Produce",
                 checkpoint: Path | None = None, epochs: int = 1, seed: int = 42,
                 workers: int = 4, cpu_threads: int = 8, promote: bool = False,
                 acceptance_mode: str = "closed_book", document_ranker_checkpoint: Path | None = None,
                 runtime_version: int = 19) -> dict:
    if runtime_version not in (19, 20):
        raise ValueError("runtime_version must be 19 or 20")
    if runtime_version == 20 and promote:
        raise ValueError("v20 acceptance promotion is not implemented.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    output_dir = Path(output_dir)
    if not output_dir.is_dir() or not (output_dir / "checkpoints").is_dir():
        raise FileNotFoundError("Existing output/checkpoints directory required.")
    dataset_bytes, acceptance_bytes = Path(dataset).read_bytes(), Path(acceptance).read_bytes()
    train, cases = load_splits(Path(dataset), Path(acceptance), acceptance_mode,
                              dataset_snapshot=dataset_bytes, acceptance_snapshot=acceptance_bytes)
    source_hashes = {"training": hashlib.sha256(dataset_bytes).hexdigest(),
                     "acceptance": hashlib.sha256(acceptance_bytes).hexdigest()}
    # Clean Slate (H1): no master baseline. No --checkpoint → fresh v20 organism.
    source_checkpoint = Path(checkpoint) if checkpoint is not None else None
    checkpoint_hash = file_sha256(source_checkpoint) if source_checkpoint is not None else None
    assistant = ZiwagoBiologicalSNNAssistant(root=ROOT, checkpoint_path=source_checkpoint, workers=workers,
                                            cpu_threads=cpu_threads, seed=seed,
                                            expected_checkpoint_sha256=checkpoint_hash,
                                            auto_load_checkpoint=(source_checkpoint is not None),
                                            runtime_version=runtime_version)
    runtime, core = assistant.genesis, assistant.organism
    if ((source_checkpoint is not None and file_sha256(source_checkpoint) != checkpoint_hash)
            or file_sha256(dataset) != source_hashes["training"]
            or file_sha256(acceptance) != source_hashes["acceptance"]):
        raise RuntimeError("A training input changed while constructing the runtime.")
    ranker_info = runtime.load_document_ranker(document_ranker_checkpoint) if document_ranker_checkpoint is not None else None
    ranker_metadata = acceptance_ranker_metadata(runtime, acceptance_mode)
    ranker_families = {family for key in ("fit_family_ids", "internal_dev_family_ids", "calibration_family_ids")
                       for family in ranker_metadata.get(key, [])}
    known_records = [item for item in train + runtime.authority.records + core.language_memory_records
                     if item.get("split", "train") == "train"]
    known_families = {item.get("family_id") for item in known_records if item.get("family_id")} | ranker_families
    if known_families & {case["family_id"] for case in cases}:
        raise ValueError("Candidate training/acceptance families overlap.")
    known_prompts = {_normal_prompt(item["prompt"]) for item in known_records}
    if any(_normal_prompt(case["prompt"]) in known_prompts
           or hashlib.sha256(_normal_prompt(case["prompt"]).encode("utf-8")).hexdigest()
           in ranker_metadata.get("exposed_prompt_hashes", []) for case in cases):
        raise ValueError("Candidate training/acceptance prompt leakage.")
    document_errors = validate_acceptance_documents(cases, acceptance_mode,
        trained_documents=[item["document"] for item in known_records if isinstance(item.get("document"), str)],
        trained_document_hashes=[item["provenance"][key] for item in known_records
                                 for key in ("context_sha256", "normalized_context_sha256")
                                 if item.get("provenance", {}).get(key)] + ranker_metadata.get("exposed_document_hashes", []))
    if document_errors:
        raise ValueError("; ".join(document_errors))
    accepted_tokens = scanned_tokens = rejected_records = 0
    schema_pairs_registered = 0
    sleep_cycles = 0
    sleep_telemetry: list = []
    physiology_deferred_records: list[dict] = []
    physiology_blocked_records: list[dict] = []
    learning_counts = {key: 0 for key in ("dendritic_pairs_trained", "dendritic_tokens_trained",
                       "response_tokens_trained", "eos_targets_trained", "spiking_token_updates",
                       "pc_targets_trained", "bank_token_updates", "bank_token_rejections", "bank_backtracking_steps",
                       "bank_limited_mask_updates", "bank_update_tokens_requested",
                       "bank_update_tokens_selected", "bank_gradient_entries_selected",
                       "bank_memory_budget_rejections", "bank_capacity_rejections",
                       "bank_objective_rejections", "bank_no_signal_rejections", "bank_other_rejections",
                       "bank_pruned_entries")}
    if runtime_version == 20:
        learning_counts["physiology_blocked_pairs"] = 0

    def physiology_report() -> dict:
        if runtime_version != 20:
            return {}
        return {"organism": core.life.telemetry(core),
                "physiology_deferred_records": len(physiology_deferred_records),
                "physiology_blocked_records": len(physiology_blocked_records),
                "physiology_deferred_or_blocked_records": len(physiology_deferred_records) + len(physiology_blocked_records),
                "physiology_records": {"deferred": physiology_deferred_records, "blocked": physiology_blocked_records},
                "sleep_telemetry": sleep_telemetry}

    diagnostics = {"before": {"train": measure_language_diagnostics(assistant, train, "train"),
                               "heldout": measure_language_diagnostics(assistant, cases, "heldout")}}
    progress_path = output_dir / f"v{runtime_version}_training_progress_{time.time_ns()}.json"
    progress = {"version": f"{runtime_version}.0", "stage": "baseline_measured",
                "source_hashes": source_hashes, "diagnostics": diagnostics}
    progress.update(physiology_report())
    _write_json_atomic(progress_path, progress)
    print(json.dumps({"stage": "baseline_measured", "progress_path": str(progress_path)}, ensure_ascii=False), flush=True)
    for epoch in range(epochs):
        for item in train:
            q_ids, r_ids = core.encode_text(item["prompt"]), core.encode_text(item["response"])
            scanned_tokens += len(q_ids) + len(r_ids)
            if (not q_ids or not 2 <= len(r_ids) < core.dendritic_plateau_readout.max_pos
                    or core.eos_token_id in r_ids):
                rejected_records += 1
                continue
            if runtime_version == 20 and not core.chemistry.is_awake:
                # One runtime-budgeted sleep advance, never a retry loop or forced wake.
                pre_pair_sleep = runtime.sleep_cycle(wave_idx=epoch + 1, requested=False)
                sleep_telemetry.append(pre_pair_sleep)
                sleep_cycles += 1
                if not core.chemistry.is_awake:
                    physiology_deferred_records.append({
                        "epoch": epoch + 1, "source": item["source"], "family_id": item["family_id"],
                        "provenance": {"path": str(Path(dataset).resolve()), "sha256": source_hashes["training"],
                                       "line": item["_source_line"]},
                        "reason": "still_asleep_after_bounded_sleep", "presented_for_training": False,
                        "bpe_tokens": len(q_ids) + len(r_ids), "sleep": pre_pair_sleep,
                        "physiology": core.chemistry.bundle()})
                    continue
            sdr = core.dendritic_plateau_readout._compute_cortical_sdr2048(q_ids + r_ids, normalize=False)
            grammar = evaluate_grammar_and_artifacts(item["response"], min_words=3)
            gate = runtime.bcm_gate.evaluate_neuromodulatory_gate(sdr, grammar_ok=bool(grammar["grammar_coherent"]))
            if not gate["accepted"]:
                rejected_records += 1
                continue
            provenance = {"path": str(Path(dataset).resolve()), "sha256": source_hashes["training"], "line": item["_source_line"]}
            for key in ("context_sha256", "normalized_context_sha256"):
                if item.get("provenance", {}).get(key):
                    provenance[key] = item["provenance"][key]
            if isinstance(item.get("document"), str):
                provenance["context_sha256"] = hashlib.sha256(item["document"].encode("utf-8")).hexdigest()
                provenance["normalized_context_sha256"] = hashlib.sha256(
                    " ".join(item["document"].casefold().split()).encode("utf-8")).hexdigest()
            runtime.authority.add_record(item["prompt"], item["response"], item["source"],
                                         family_id=item["family_id"], provenance=provenance)
            graph_source = f'{item["source"]}|{provenance["path"]}|sha256:{provenance["sha256"]}|line:{provenance["line"]}'
            runtime.world_graph.extract_and_consolidate_sentence(item["response"], source=graph_source, status="asserted")
            if not runtime.authority.ground(item["prompt"], item["response"])["grounded"]:
                rejected_records += 1
                continue
            trained = runtime.train_wave_with_neuromodulator_reward(
                [TokenizedDialogPair(item["source"], q_ids, r_ids, quality_reward_hint=2.0)], wave_idx=epoch + 1)
            for key in learning_counts:
                learning_counts[key] += trained[key]
            if runtime_version == 20 and trained["physiology_blocked_pairs"]:
                # Exposure and any partial updates are not a learned dendritic pair.
                physiology_blocked_records.append({
                    "epoch": epoch + 1, "source": item["source"], "family_id": item["family_id"],
                    "provenance": provenance, "reason": "physiology_blocked_training",
                    "presented_for_training": True, "bpe_tokens": len(q_ids) + len(r_ids),
                    "learning_counts": {key: trained[key] for key in learning_counts},
                    "physiology": core.chemistry.bundle()})
            if trained["dendritic_pairs_trained"]:
                runtime.emergent_ontology.register_schema_pair(item["prompt"], item["response"])
                schema_pairs_registered += 1
            if runtime_version == 19 or trained["dendritic_pairs_trained"]:
                accepted_tokens += len(q_ids) + len(r_ids)
            # Natural sleep is advanced without forcing v20 to wake or finish a cycle.
            if not core.chemistry.is_awake:
                natural_sleep = (runtime.sleep_cycle(wave_idx=epoch + 1, requested=False)
                                 if runtime_version == 20 else runtime.sleep_cycle(wave_idx=epoch + 1))
                sleep_telemetry.append(natural_sleep)
                sleep_cycles += 1
        # Preserve v19 epoch consolidation; v20 only advances an already sleeping organism.
        epoch_sleep = (runtime.sleep_cycle(wave_idx=epoch + 1, requested=False)
                       if runtime_version == 20 else runtime.sleep_cycle(wave_idx=epoch + 1))
        sleep_telemetry.append(epoch_sleep)
        # An awake v20 no-op is reported, but is not counted as a sleep cycle.
        sleep_cycles += int(runtime_version == 19 or epoch_sleep["elapsed_ms"] > 0.0)
        progress.update({"stage": "epoch_finished", "epochs_completed": epoch + 1,
                         "accepted_bpe_tokens": accepted_tokens, "rejected_records": rejected_records,
                         "ontology_schema_pairs_registered": schema_pairs_registered,
                         "sleep_cycles": sleep_cycles,
                         "last_sleep": {k: v for k, v in epoch_sleep.items() if k != "rem"},
                         "rem": epoch_sleep.get("rem"),
                         "learning_counts": learning_counts})
        _write_json_atomic(progress_path, progress)
        print(json.dumps({"stage": "epoch_finished", "epoch": epoch + 1, "learning_counts": learning_counts}), flush=True)
    diagnostics["after"] = {"train": measure_language_diagnostics(assistant, train, "train"),
                             "heldout": measure_language_diagnostics(assistant, cases, "heldout")}
    if ((source_checkpoint is not None and file_sha256(source_checkpoint) != checkpoint_hash)
            or file_sha256(dataset) != source_hashes["training"]
            or file_sha256(acceptance) != source_hashes["acceptance"]):
        raise RuntimeError("A training input changed before candidate saving.")
    if ranker_info is not None and file_sha256(document_ranker_checkpoint) != ranker_info["sha256"]:
        raise RuntimeError("Document ranker checkpoint changed during training.")
    candidate_path = output_dir / "checkpoints" / f"ziwago_v{runtime_version}_candidate_{time.time_ns()}.pt"
    saved = runtime.save_candidate(candidate_path, extra_metadata={"source_hashes": source_hashes, "epochs": epochs,
                                                                "source_checkpoint_sha256": checkpoint_hash})
    # Acceptance is performed after loading the written candidate, not on a different in-memory organism.
    if runtime_version == 20:
        runtime.load_v20_checkpoint(candidate_path, expected_checkpoint_sha256=saved["sha256"])
    else:
        runtime.load_v19_checkpoint(candidate_path, expected_checkpoint_sha256=saved["sha256"])
    report = evaluate_frozen_acceptance(assistant, train, cases, acceptance_mode)
    report.update({"checkpoint_sha256": saved["sha256"], "source_hashes": source_hashes,
                   "train_family_hash": stable_hash(sorted({item["family_id"] for item in train})),
                   "full_corpus_training_certified": False,
                   "scanned_bpe_tokens": scanned_tokens, "accepted_bpe_tokens": accepted_tokens,
                   "rejected_records": rejected_records})
    if file_sha256(candidate_path) != saved["sha256"]:
        raise RuntimeError("The candidate changed during acceptance evaluation.")
    if ((source_checkpoint is not None and file_sha256(source_checkpoint) != checkpoint_hash)
            or file_sha256(dataset) != source_hashes["training"]
            or file_sha256(acceptance) != source_hashes["acceptance"]):
        raise RuntimeError("A training input changed during candidate evaluation.")
    result = {"version": f"{runtime_version}.0", "candidate": saved, "acceptance": report, "legacy_master_unchanged": True,
              "source_checkpoint_sha256": checkpoint_hash, "source_checkpoint_unchanged": True,
              "source_files_unchanged": True,
              "language_diagnostics": diagnostics, "learning_counts": learning_counts,
              "ontology_schema_pairs_registered": schema_pairs_registered,
              "sleep_cycles": sleep_cycles,
              "sleep_telemetry": sleep_telemetry,
              "progress_path": str(progress_path)}
    if promote:
        result["promotion"] = accept_candidate(candidate_path, report,
                                                output_dir / "checkpoints/ziwago_v19_0_accepted.pt")
    report_path = output_dir / f"v{runtime_version}_acceptance_{time.time_ns()}.json"
    result["report_path"] = str(report_path)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    progress.update({"stage": "completed", "report_path": str(report_path), "candidate_sha256": saved["sha256"],
                     "acceptance_passed": bool(report.get("passed"))})
    _write_json_atomic(progress_path, progress)
    return result


def main(default_version: int = 19) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-version", type=int, choices=(19, 20), default=default_version)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Produce")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--acceptance-mode", choices=("closed_book", "open_book"), default="closed_book")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--document-ranker", type=Path)
    args = parser.parse_args()
    if args.evaluate_only:
        if args.checkpoint is None or args.promote:
            parser.error("--evaluate-only requires --checkpoint and forbids --promote.")
        result = run_evaluation(args.dataset, args.acceptance, args.checkpoint, args.output_dir,
                                args.seed, workers=args.workers, cpu_threads=args.cpu_threads,
                                acceptance_mode=args.acceptance_mode, document_ranker_checkpoint=args.document_ranker,
                                runtime_version=args.runtime_version)
    else:
        result = run_training(args.dataset, args.acceptance, args.output_dir, args.checkpoint,
                              args.epochs, args.seed, workers=args.workers, cpu_threads=args.cpu_threads,
                              promote=args.promote, acceptance_mode=args.acceptance_mode,
                              document_ranker_checkpoint=args.document_ranker,
                              runtime_version=args.runtime_version)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["acceptance"].get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
