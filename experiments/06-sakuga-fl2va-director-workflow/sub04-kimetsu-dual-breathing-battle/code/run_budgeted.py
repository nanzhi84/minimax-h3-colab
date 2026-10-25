"""Reuse the Studio lifecycle with a dedicated 30.0 CU budget and stop watchdog for Sub-Experiment 04."""

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
OUT = EXP / "outputs"
PRIVATE = REPO / "work" / "exp06-sub04-budget-private.json"
LIMIT = 30.0
RESERVE = 2.0
STOP_RATE_PER_HOUR = 9.0
SESSION = "h3-exp06-sub04-kimetsu"
sys.path.insert(0, str(REPO))
from studio.api import colab_batch_runner as runner


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def write_private(state: dict) -> None:
    PRIVATE.parent.mkdir(parents=True, exist_ok=True)
    temp = PRIVATE.with_suffix(".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(state, indent=2, fp=stream)
    temp.replace(PRIVATE)


def account() -> dict:
    return runner.fetch_ccu_info(runner.get_fresh_credentials())["response_body"]


def reconcile(state: dict, body: dict) -> dict:
    measured = max(0.0, state["initial_balance"] - float(body["currentBalance"]))
    summed = sum(float(b.get("actual_delta_cu", 0)) for b in state["batches"])
    state["spent_cu"] = round(max(measured, summed, state.get("spent_cu", 0)), 6)
    for batch in state["batches"]:
        if "finished_at" in batch:
            wall = (
                dt.datetime.fromisoformat(batch["finished_at"])
                - dt.datetime.fromisoformat(batch["started_at"])
            ).total_seconds()
            batch["wall_upper_cu"] = round(wall * STOP_RATE_PER_HOUR / 3600, 6)
    state["budget_committed_cu"] = round(
        max(state["spent_cu"], sum(b.get("wall_upper_cu", 0) for b in state["batches"])), 6
    )
    state["last_reconciled_at"] = now()
    state["active_assignments"] = int(body.get("assignmentsCount", 0))
    write_private(state)
    public = {
        k: state[k]
        for k in [
            "started_at",
            "spent_cu",
            "budget_committed_cu",
            "last_reconciled_at",
            "active_assignments",
            "batches",
        ]
    }
    public.update(
        {
            "limit_cu": LIMIT,
            "reserve_cu": RESERVE,
            "configured_stop_rate_cu_per_hour": STOP_RATE_PER_HOUR,
            "remaining_cu": round(LIMIT - state["budget_committed_cu"], 6),
            "within_budget": state["budget_committed_cu"] <= LIMIT,
            "accounting_note": (
                "spent_cu is observed account debit; budget_committed_cu also reserves a "
                "conservative wall-time upper bound to cover delayed billing."
            ),
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "budget_ledger.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n")
    return public


def owns_cached_session() -> bool:
    if not runner.SESSIONS_PATH.exists():
        return False
    return SESSION in json.loads(runner.SESSIONS_PATH.read_text())


def stop_owned() -> None:
    if owns_cached_session():
        result = subprocess.run(
            [runner.COLAB_BIN, "stop", "-s", SESSION],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Stopping the owned session failed; inspect colab sessions")


def run(manifest: Path, max_batch_cu: float, dry_run: bool) -> None:
    if dry_run:
        runner.run_colab_batch(
            manifest, outputs_root=OUT / "batches", session_name=SESSION, dry_run=True
        )
        return
    lock_path = PRIVATE.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(PRIVATE.read_text())
        before = account()
        current = reconcile(state, before)
        if current["active_assignments"]:
            raise RuntimeError("Active assignments exist; do not overlap another GPU session")
        cap = min(max_batch_cu, LIMIT - state["budget_committed_cu"] - RESERVE)
        if cap < 0.5:
            raise RuntimeError("Insufficient experiment budget after the shutdown reserve")
        deadline_sec = max(1.0, cap / (STOP_RATE_PER_HOUR / 3600) - 90)
        record = {
            "manifest": str(manifest.resolve().relative_to(REPO)),
            "started_at": now(),
            "status": "RUNNING",
            "cost_cap_cu": round(cap, 4),
            "configured_stop_rate_cu_per_hour": STOP_RATE_PER_HOUR,
            "watchdog_seconds": round(deadline_sec, 2),
        }
        state["batches"].append(record)
        reconcile(state, before)
        done = threading.Event()
        expired = threading.Event()
        start = time.monotonic()

        def watchdog() -> None:
            while not done.wait(3):
                if time.monotonic() - start > deadline_sec:
                    expired.set()
                    if owns_cached_session():
                        for attempt in range(3):
                            try:
                                stop_owned()
                                break
                            except (RuntimeError, subprocess.SubprocessError) as error:
                                print(
                                    f"WATCHDOG STOP FAILURE {attempt + 1}/3: {type(error).__name__}",
                                    flush=True,
                                )
                                if attempt == 2:
                                    record["watchdog_stop_failed"] = True
                                    return
                                done.wait(3)

        thread = threading.Thread(target=watchdog, daemon=True)
        thread.start()
        print(
            json.dumps(
                {
                    "budget_remaining_cu": current["remaining_cu"],
                    "batch_cap_cu": cap,
                    "watchdog_seconds": round(deadline_sec, 1),
                }
            ),
            flush=True,
        )
        try:
            runner.run_colab_batch(manifest, outputs_root=OUT / "batches", session_name=SESSION)
            record["status"] = "COMPLETE"
        except BaseException as error:
            record["status"] = "FAILED"
            record["error_type"] = type(error).__name__
            raise
        finally:
            try:
                stop_owned()
            finally:
                done.set()
                thread.join(timeout=65)
                record["finished_at"] = now()
                record["watchdog_expired"] = expired.is_set()
                after = account()
                record["actual_delta_cu"] = round(
                    max(0.0, float(before["currentBalance"]) - float(after["currentBalance"])), 6
                )
                public = reconcile(state, after)
                print("EXPERIMENT_BUDGET:" + json.dumps(public, ensure_ascii=False), flush=True)
                if public["active_assignments"]:
                    raise RuntimeError("GPU assignment remains active after shutdown")
        if expired.is_set():
            raise RuntimeError("Experiment batch watchdog expired")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batch-cu", type=float, default=3.5)
    args = parser.parse_args()
    if args.init:
        if PRIVATE.exists():
            raise FileExistsError("Budget already initialized; use --status to preserve prior spending")
        body = account()
        if body.get("assignmentsCount", 0):
            raise RuntimeError("Start experiment only when existing assignments are zero")
        state = {
            "started_at": now(),
            "initial_balance": float(body["currentBalance"]),
            "spent_cu": 0.0,
            "batches": [],
        }
        print(json.dumps(reconcile(state, body), ensure_ascii=False))
    elif args.status:
        print(json.dumps(reconcile(json.loads(PRIVATE.read_text()), account()), ensure_ascii=False, indent=2))
    elif args.manifest:
        run(args.manifest.resolve(), args.max_batch_cu, args.dry_run)
    else:
        parser.error("Choose --init, --status or --manifest")


if __name__ == "__main__":
    main()
