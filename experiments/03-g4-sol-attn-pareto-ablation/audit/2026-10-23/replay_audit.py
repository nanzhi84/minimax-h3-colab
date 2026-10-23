#!/usr/bin/env python3
"""Reproduce CPU source-audit calculations; this is not a GPU/E2E benchmark."""

import ast
import hashlib
import json
import math
import platform
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
EVIDENCE = HERE / "source_evidence.json"
METRICS = EXPERIMENT / "outputs/g4_sol_attn_ablation_metrics.json"


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def main():
    evidence = json.loads(EVIDENCE.read_text())
    metrics = json.loads(METRICS.read_text())
    source = next(s for s in evidence["sources"] if s["id"] == "community_schedule")
    snippet = source["excerpts"][0]["code"]
    # The inspected upstream class has no imports or GPU dependencies.
    node = ast.parse(snippet).body[0]
    if not isinstance(node, ast.ClassDef) or node.name != "_TauSchedule":
        raise ValueError("Expected the audited upstream _TauSchedule class")
    namespace = {"math": math, "__builtins__": {
        "__build_class__": __build_class__, "float": float, "min": min, "max": max,
    }, "__name__": "audited_upstream_schedule"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), source["url"], "exec"), namespace)
    schedule = namespace["_TauSchedule"](1.5, 0.8, "cosine", 1.0, 0.0)

    # ComfyUI simple/8 picks base t = 1, 7/8, ... 1/8 from its 1000-step grid.
    # These inputs and intermediate products are exact in FP32; the division rounds.
    rows = []
    for step in range(8):
        t = 1.0 - step / 8
        sigma = fp32(12.0 * t / (1.0 + 11.0 * t))
        original = schedule.progress(sigma)
        corrected = min(max(1.0 - sigma / (12.0 - 11.0 * sigma), 0.0), 1.0)
        rows.append({"step_1_based": step + 1, "sigma_fp32": sigma,
                     "stock_progress": original, "unshifted_progress": corrected,
                     "stock_dense": original < 0.2, "fixed_dense": corrected < 0.2})

    runs = {r["run_id"]: r for r in metrics["runs"]}
    micro = metrics["microbenchmark"]
    projected_chunks = math.ceil(metrics["config"]["joint_sequence_tokens"] / 4096)
    video_hashes = {str(i): {"file": runs[i]["mp4_file"],
                            "sha256": sha256(EXPERIMENT / "outputs" / runs[i]["mp4_file"])}
                    for i in (2, 4, 8, 12)}
    pairs = []
    for before, after in ((5, 7), (6, 8), (2, 4), (8, 12)):
        pairs.append({"before": before, "after": after,
                      "denoise_saved_sec": round(runs[before]["denoise_sec"] - runs[after]["denoise_sec"], 4),
                      "e2e_saved_sec": round(runs[before]["e2e_sec"] - runs[after]["e2e_sec"], 4),
                      "Q_total_delta": round(runs[after]["Q_total"] - runs[before]["Q_total"], 4)})

    checks = {
        "source_schedule_stock_dense_7_of_8": sum(r["stock_dense"] for r in rows) == 7,
        "inverse_shift_dense_2_of_8": sum(r["fixed_dense"] for r in rows) == 2,
        "replay_matches_archived_schedule_gates": all(
            r["stock_dense"] == old["stock_is_dense"] and r["fixed_dense"] == old["fixed_is_dense"]
            for r, old in zip(rows, micro["discovery2_schedule_verification"]["steps"])),
        "source_arithmetic_matches_archived_qkv_calls": (
            projected_chunks == micro["discovery1_block_benchmark"]["chunked_warm_qkv_calls_per_block"] == 11
            and 2 * projected_chunks == micro["discovery1_block_benchmark"]["chunked_cold_qkv_calls_per_block"] == 22),
        "run11_archived_100_dit_dense_300_sparse": (
            runs[11]["kernel_calls"]["sage"] == 102 and runs[11]["kernel_calls"]["triton_sol"] == 300),
        "run02_run04_existing_mp4_bytes_equal": video_hashes["2"]["sha256"] == video_hashes["4"]["sha256"],
        "run08_run12_existing_mp4_bytes_differ": video_hashes["8"]["sha256"] != video_hashes["12"]["sha256"],
    }
    result = {
        "scope": "CPU source replay and existing artifact verification only; no GPU timing, CUDA numerical test, or fresh E2E generation.",
        "python": platform.python_version(), "platform": platform.system(),
        "inputs": {"source_evidence_sha256": sha256(EVIDENCE), "experiment_metrics_sha256": sha256(METRICS)},
        "checks": checks, "passed": sum(checks.values()), "total": len(checks),
        "schedule": rows, "existing_video_hashes": video_hashes,
        "historical_single_request_pairs": pairs,
        "full_bf16_qkv_gib": round(42058 * 3 * 56 * 128 * 2 / 2 ** 30, 4),
        "chunk_bf16_qkv_gib": round(4096 * 3 * 56 * 128 * 2 / 2 ** 30, 4),
    }
    output = HERE / "replay_result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit("Source-audit evidence mismatch; see replay_result.json")


if __name__ == "__main__":
    main()
