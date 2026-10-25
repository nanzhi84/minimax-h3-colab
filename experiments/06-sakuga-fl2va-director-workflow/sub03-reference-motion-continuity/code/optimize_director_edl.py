# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Optimize multi-seed shot selection and monotonic frame-index schedules against the reference video."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from analyze_motion import decode

EXP = Path(__file__).resolve().parents[1]
SOURCE = EXP / "outputs" / "reference" / "wuy0uer_7664781155287239972.mp4"


def extract_features(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute blurred color (64x36), structural gradient (128x72), and full-frame step motion."""
    colors, grads = [], []
    for f in frames:
        small = cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(small, (5, 5), 1.0)
        colors.append(blur)
        gray = cv2.cvtColor(cv2.resize(f, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grads.append(np.hypot(gx, gy))
    c_arr = np.stack(colors, axis=0)
    g_arr = np.stack(grads, axis=0)
    diffs = np.zeros(len(frames), dtype=np.float32)
    if len(frames) > 1:
        diffs[1:] = np.mean(np.abs(c_arr[1:] - c_arr[:-1]), axis=(1, 2, 3))
        diffs[0] = diffs[1]
    return c_arr, g_arr, diffs


def solve_monotonic_schedule(
    ref_c: np.ndarray,
    ref_g: np.ndarray,
    ref_mot: np.ndarray,
    cand_c: np.ndarray,
    cand_g: np.ndarray,
    target_count: int,
    prev_last_c: np.ndarray | None = None,
) -> tuple[list[int], float, float]:
    """Find strictly increasing indices 0 <= i_0 < ... < i_{M-1} < N minimizing reference & pacing cost."""
    n = len(cand_c)
    m = target_count
    if m > n:
        raise ValueError(f"Cannot select {m} unique frames from {n} frames")

    c_dist = np.mean(np.abs(ref_c[:, None] - cand_c[None, :]), axis=(2, 3, 4))
    g_dist = np.mean(np.abs(ref_g[:, None] - cand_g[None, :]), axis=(2, 3))
    unary = 0.72 * c_dist + 0.28 * g_dist

    ideal_step = (n - 2) / max(1, m - 1)
    max_step = max(6, int(np.ceil((n - 1) / max(1, m - 1) * 2.4)))
    dp = np.full((m, n), 1e9, dtype=np.float64)
    parent = np.full((m, n), -1, dtype=np.int32)

    # Anchor start in 0..1 and end in n-2..n-1 (37..38) so no boundary transition frames are skipped
    max_start = 0 if prev_last_c is None else min(1, n - m)
    for i in range(max_start + 1):
        start_bias = 0.002 * i
        if prev_last_c is not None:
            seam_diff = float(np.mean(np.abs(cand_c[i] - prev_last_c)))
            if seam_diff * 255.0 < 1.2:
                start_bias += 0.020
            else:
                start_bias += 0.60 * seam_diff
        dp[0, i] = unary[0, i] + start_bias

    for t in range(1, m):
        rem = m - 1 - t
        i_min = t
        i_max = n - 1 - rem
        for i in range(i_min, i_max + 1):
            p_min = max(t - 1, i - max_step)
            p_max = i - 1
            best_val = 1e9
            best_p = -1
            for p in range(p_min, p_max + 1):
                step = i - p
                step_cost = 0.008 * ((step - ideal_step) ** 2)
                frame_delta = float(np.mean(np.abs(cand_c[i] - cand_c[p])))
                stall_cost = 0.025 if (ref_mot[t] > 0.015 and frame_delta < 0.006) else 0.0
                val = dp[t - 1, p] + unary[t, i] + step_cost + stall_cost
                if val < best_val:
                    best_val = val
                    best_p = p
            dp[t, i] = best_val
            parent[t, i] = best_p

    # End in n-2..n-1 (37..38) so the shot reaches its target keyframe f_end before handing off
    min_end = max(m - 1, n - 2)
    end_costs = dp[m - 1, min_end:n].copy()
    best_end = min_end + int(np.argmin(end_costs))
    total_cost = float(dp[m - 1, best_end])

    schedule = [0] * m
    cur = best_end
    for t in range(m - 1, -1, -1):
        schedule[t] = int(cur)
        cur = int(parent[t, cur])

    raw_mae = float(np.mean([c_dist[t, schedule[t]] for t in range(m)])) * 255.0
    return schedule, total_cost / m, raw_mae


def collect_candidates(batches_filter: list[tuple[str, str]]) -> dict[str, list[dict]]:
    """Index all available candidate takes by shot_id across completed batches."""
    by_shot: dict[str, list[dict]] = {}
    for round_name, batch_rel in batches_filter:
        batch_dir = EXP / batch_rel
        metrics_path = batch_dir / "studio_batch_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        for job in metrics["jobs"]:
            mp4 = batch_dir / job["mp4_file"]
            if not mp4.exists():
                continue
            by_shot.setdefault(job["shot_id"], []).append({
                "round": round_name,
                "batch": batch_rel,
                "job_id": job["job_id"],
                "seed": job["seed"],
                "mp4": mp4,
                "prod_quality_score": float(job.get("prod_quality_score", 80.0)),
                "S_fl2v_pixel": float(job.get("S_fl2v_pixel", 90.0)),
            })
    return by_shot


def build_active_shot_specs(by_shot: dict[str, list[dict]]) -> list[dict]:
    """Assemble the 36-shot sub-beat chain when round06 splits are present, else 31-shot chain."""
    base_shots = json.loads((EXP / "code" / "reproduction_round04.json").read_text())["shots"]
    r6_path = EXP / "code" / "reproduction_round06.json"
    r6_map: dict[str, list[dict]] = {}
    if r6_path.exists():
        r6_shots = json.loads(r6_path.read_text())["shots"]
        replacements = {
            "m05_leap_to_moon_ring": ["m05a_moon_silhouette_leap", "m05b_moon_crescent_ring"],
            "m10_moon_radial_burst": ["m10a_banner_to_moon_orb", "m10b_moon_orb_to_burst"],
            "m11_burst_to_closeup": ["m11a_burst_to_roof_peak", "m11b_roof_peak_to_closeup"],
            "m13_halo_drop_to_roof": ["m13a_halo_to_white_plunge", "m13b_white_plunge_to_roof"],
            "m15_white_deck_charge": ["m15a_roof_to_close_slash", "m15b_close_slash_to_white_deck"],
        }
        r6_by_id = {s["id"]: s for s in r6_shots}
        for orig_id, sub_ids in replacements.items():
            if all(sid in by_shot and sid in r6_by_id for sid in sub_ids):
                r6_map[orig_id] = [r6_by_id[sid] for sid in sub_ids]

    final_shots = []
    for s in base_shots:
        if s["id"] in r6_map:
            final_shots.extend(r6_map[s["id"]])
        else:
            final_shots.append(s)
    return final_shots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-edls", action="store_true", help="Write optimized edl_director_29s83.json and edl_director_tight.json")
    args = parser.parse_args()

    ref_frames = decode(SOURCE)
    ref_c, ref_g, ref_mot = extract_features(ref_frames)

    batches = [
        ("round04a", "outputs/reproduction/batches/exp06_sub03_round04a/1"),
        ("round04b", "outputs/reproduction/batches/exp06_sub03_round04b/1"),
        ("round05", "outputs/reproduction/batches/exp06_sub03_round05/1"),
        ("round06", "outputs/reproduction/batches/exp06_sub03_round06/1"),
    ]
    by_shot = collect_candidates(batches)
    shots = build_active_shot_specs(by_shot)

    moon_shots = [s for s in shots if s["id"].startswith("m")]
    autumn_shots = [s for s in shots if s["id"].startswith("a")]

    def build_chain_parts(chain_shots: list[dict], scale: float = 1.0, fade_last: int = 0) -> tuple[list[dict], list[dict]]:
        parts, diagnostics = [], []
        prev_last_c = None
        for idx, shot in enumerate(chain_shots):
            sid = shot["id"]
            is_last = idx == len(chain_shots) - 1
            f_start, f_end = int(shot["first"]), int(shot["last"])
            c0 = round(f_start * 24 * scale / 30)
            c1 = round(f_end * 24 * scale / 30)
            target_frames = max(4, c1 - c0)
            if is_last and fade_last > 0:
                target_frames += fade_last

            # Half-open interval (f_start, f_end] for interior shots so each shot reaches f_end without duplicating f_start
            if is_last:
                end_f = min(894, f_end + round(fade_last * 30 / 24))
                ref_idx = np.linspace(f_start, end_f, target_frames + (1 if idx > 0 else 0))[1 if idx > 0 else 0:].round().astype(int)
            elif idx == 0:
                ref_idx = np.linspace(f_start, f_end, target_frames).round().astype(int)
            else:
                ref_idx = np.linspace(f_start, f_end, target_frames + 1)[1:].round().astype(int)
            r_c = ref_c[ref_idx]
            r_g = ref_g[ref_idx]
            r_m = ref_mot[ref_idx]

            cands = by_shot.get(sid, [])
            if not cands:
                raise RuntimeError(f"No completed candidates found for shot {sid}")

            best_choice = None
            all_scores = []
            for cand in cands:
                frames = decode(cand["mp4"])
                c_c, c_g, _ = extract_features(frames)
                sched, dp_cost, mae_255 = solve_monotonic_schedule(
                    r_c, r_g, r_m, c_c, c_g, target_frames, prev_last_c=prev_last_c
                )
                seam_cost = 0.0
                if prev_last_c is not None:
                    seam_cost = float(np.mean(np.abs(c_c[sched[0]] - prev_last_c))) * 255.0
                # Prefer round06 refined takes when scores are close
                round_bonus = 1.5 if cand["round"] == "round06" else 0.0
                combined = mae_255 + 0.15 * seam_cost - 0.08 * cand["prod_quality_score"] - round_bonus
                entry = {
                    "cand": cand,
                    "schedule": sched,
                    "dp_cost": round(dp_cost, 5),
                    "ref_mae_255": round(mae_255, 2),
                    "seam_mae_255": round(seam_cost, 2),
                    "combined_cost": round(combined, 3),
                    "last_c": c_c[sched[-1]],
                }
                all_scores.append(entry)
                if best_choice is None or entry["combined_cost"] < best_choice["combined_cost"]:
                    best_choice = entry

            assert best_choice is not None
            prev_last_c = best_choice["last_c"]
            chosen = best_choice["cand"]
            sched = best_choice["schedule"]
            part = {
                "batch": chosen["batch"],
                "job_id": chosen["job_id"],
                "start_frame": sched[0],
                "end_frame_exclusive": sched[-1] + 1,
                "selected_frames": sched,
                "target_frames": target_frames,
                "reference_window": f"f{f_start:03d}-f{f_end:03d}",
                "reason": (
                    f"{shot['title']}：DP单调选帧 {sched[0]}..{sched[-1]} -> {target_frames}帧 "
                    f"(ref_mae={best_choice['ref_mae_255']}, seam={best_choice['seam_mae_255']}, seed={chosen['seed']})"
                ),
            }
            if is_last and fade_last > 0:
                part["fade_out_frames"] = fade_last
            parts.append(part)
            diagnostics.append({
                "shot_id": sid,
                "title": shot["title"],
                "window": f"f{f_start:03d}-f{f_end:03d}",
                "target_frames": target_frames,
                "selected_job": chosen["job_id"],
                "selected_seed": chosen["seed"],
                "ref_mae_255": best_choice["ref_mae_255"],
                "seam_mae_255": best_choice["seam_mae_255"],
                "schedule_span": f"{sched[0]}..{sched[-1]} ({len(sched)}f)",
                "candidates": [
                    {
                        "job_id": e["cand"]["job_id"],
                        "seed": e["cand"]["seed"],
                        "ref_mae_255": e["ref_mae_255"],
                        "seam_mae_255": e["seam_mae_255"],
                        "prod_quality_score": e["cand"]["prod_quality_score"],
                        "combined_cost": e["combined_cost"],
                    }
                    for e in all_scores
                ],
            })
        return parts, diagnostics

    moon_parts, moon_diag = build_chain_parts(moon_shots, scale=1.0, fade_last=0)
    autumn_parts, autumn_diag = build_chain_parts(autumn_shots, scale=1.0, fade_last=3)

    moon_total = sum(p["target_frames"] for p in moon_parts)
    autumn_total = sum(p["target_frames"] for p in autumn_parts)
    mid_black = 11
    end_black = 716 - (moon_total + mid_black + autumn_total)
    if end_black < 1:
        raise ValueError(f"Invalid end_black frames: {end_black}")

    edl_29s83 = {
        "name": "selected_reproduction",
        "target_total_frames": 716,
        "description": (
            f"最终选片版（29.833s / 716帧）：基于{len(shots)}个逐拍对齐短窗口（38个原片关键帧与亚拍锚点，单镜39帧）与多Seed抽卡，"
            "通过参考片引导的半开区间单调动态规划（Monotonic DP）逐帧对齐原片动作节奏并切掉首尾惯性停顿。"
        ),
        "parts": [
            *moon_parts,
            {
                "black": True,
                "target_frames": mid_black,
                "reference_window": "f439-f451",
                "reason": "对齐原片中段13帧@30fps（0.433s）显式黑场转场。",
            },
            *autumn_parts,
            {
                "black": True,
                "target_frames": end_black,
                "reference_window": "f867-f894",
                "reason": "对齐原片尾段黑场收束，确保全片精确716帧（29.833s @ 24fps）。",
            },
        ],
    }

    moon_tight, _ = build_chain_parts(moon_shots, scale=0.86, fade_last=0)
    autumn_tight, _ = build_chain_parts(autumn_shots, scale=0.86, fade_last=3)
    tight_mid_black = 2
    tight_end_black = 6
    tight_total = sum(p["target_frames"] for p in moon_tight) + tight_mid_black + sum(p["target_frames"] for p in autumn_tight) + tight_end_black

    edl_tight = {
        "name": "selected_reproduction_tight",
        "target_total_frames": tight_total,
        "description": (
            f"节奏强化导演版（{tight_total / 24:.2f}s / {tight_total}帧）：在{len(shots)}镜逐拍对齐链基础上，"
            "使用DP只抽取每镜最高速动作核（0.86×时长压缩）并压缩中段与尾段黑场留白。"
        ),
        "parts": [
            *moon_tight,
            {
                "black": True,
                "target_frames": tight_mid_black,
                "reference_window": "f439-f451",
                "reason": "紧凑版保留2帧黑场呼吸切口。",
            },
            *autumn_tight,
            {
                "black": True,
                "target_frames": tight_end_black,
                "reference_window": "f870-f894",
                "reason": "紧凑版尾段淡出后保留6帧黑场收尾。",
            },
        ],
    }

    all_diag = moon_diag + autumn_diag
    worst = sorted(all_diag, key=lambda d: d["ref_mae_255"], reverse=True)
    summary = {
        "total_shots": len(all_diag),
        "mean_ref_mae_255": round(float(np.mean([d["ref_mae_255"] for d in all_diag])), 2),
        "max_ref_mae_255": round(float(np.max([d["ref_mae_255"] for d in all_diag])), 2),
        "mean_seam_mae_255": round(float(np.mean([d["seam_mae_255"] for d in all_diag if d["seam_mae_255"] > 0])), 2),
        "top_5_highest_mae_shots": worst[:5],
        "shots": all_diag,
    }
    out_report = EXP / "outputs" / "reproduction" / "dp_optimization_report.json"
    out_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    if args.write_edls:
        (EXP / "code" / "edl_director_29s83.json").write_text(json.dumps(edl_29s83, ensure_ascii=False, indent=2) + "\n")
        (EXP / "code" / "edl_director_tight.json").write_text(json.dumps(edl_tight, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps({
        "status": "OK",
        "total_shots": len(all_diag),
        "mean_ref_mae_255": summary["mean_ref_mae_255"],
        "max_ref_mae_255": summary["max_ref_mae_255"],
        "mean_seam_mae_255": summary["mean_seam_mae_255"],
        "edl_29s83_frames": edl_29s83["target_total_frames"],
        "edl_tight_frames": edl_tight["target_total_frames"],
        "top_5_highest_mae": [(w["shot_id"], w["ref_mae_255"], w["selected_job"]) for w in worst[:5]],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
