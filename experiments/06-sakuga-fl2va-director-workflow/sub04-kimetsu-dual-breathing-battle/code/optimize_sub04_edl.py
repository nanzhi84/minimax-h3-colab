# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "opencv-python-headless==5.0.0.93", "pillow==13.0.0"]
# ///
"""Universal Sakuga Gacha selector and Velocity-Regularized Arc-Length Monotonic DP frame scheduler for Sub04."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from review_candidates import decode

EXP = Path(__file__).resolve().parents[1]
ALIGNED_DIR = EXP / "outputs" / "keyframes_aligned"


def extract_clip_features(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 64x36 blurred color [0,1], 128x72 Sobel gradient [0,1], and normalized sharpness."""
    colors, grads, sharps = [], [], []
    for f in frames:
        small = cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        colors.append(cv2.GaussianBlur(small, (5, 5), 1.0))
        gray = (
            cv2.cvtColor(cv2.resize(f, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(
                np.float32
            )
            / 255.0
        )
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grads.append(np.hypot(gx, gy))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        sharps.append(lap_var)
    s_arr = np.array(sharps, dtype=np.float32)
    s_norm = (s_arr - s_arr.min()) / (float(np.ptp(s_arr)) + 1e-6)
    return np.stack(colors, axis=0), np.stack(grads, axis=0), s_norm


def load_kf_feature(kf_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = ALIGNED_DIR / f"{kf_id}.png"
    rgb = np.asarray(Image.open(path).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    c, g, _ = extract_clip_features([bgr])
    return c[0], g[0]


def solve_velocity_regularized_schedule(
    cand_c: np.ndarray,
    cand_g: np.ndarray,
    cand_sharp: np.ndarray,
    first_kf_c: np.ndarray,
    last_kf_c: np.ndarray,
    target_count: int,
    prev_last_c: np.ndarray | None = None,
    sub_range: tuple[int, int] | None = None,
) -> tuple[list[int], float, float, float]:
    """Select strictly increasing native indices using Arc-Length Motion Progress DP and boundary de-duplication."""
    lo, hi = sub_range if sub_range is not None else (0, len(cand_c))
    sub_c = cand_c[lo:hi]
    sub_s = cand_sharp[lo:hi]
    n = len(sub_c)
    m = target_count
    if m > n:
        raise ValueError(f"Cannot select {m} unique frames from {n} frames")

    step_diffs = np.mean(np.abs(sub_c[1:] - sub_c[:-1]), axis=(1, 2, 3)) * 255.0
    median_step = float(np.median(step_diffs)) if len(step_diffs) > 0 else 4.0

    # Cumulative motion arc-length progress [0, 1] along the actual clip trajectory
    cum_mot = np.concatenate([[0.0], np.cumsum(np.maximum(step_diffs, 0.25))])
    cum_mot = cum_mot / max(1e-6, float(cum_mot[-1]))
    alphas = np.linspace(0.0, 1.0, m)

    # Determine valid start range: if prev_last_c is present and frame 0 duplicates prev_last_c (< 3.5/255),
    # skip frame 0 (and frame 1 if frame 0->1 is an on-twos hold) to eliminate the boundary freeze.
    if prev_last_c is None:
        min_start, max_start = 0, min(1, n - m)
    else:
        dup_diff_0 = float(np.mean(np.abs(sub_c[0] - prev_last_c))) * 255.0
        if dup_diff_0 < 3.5 and n - m >= 1:
            min_start = 1
            max_start = min(2, n - m)
        else:
            min_start = 0
            max_start = min(2, n - m)

    ideal_step = (n - 1 - min_start) / max(1, m - 1)
    max_step = max(3, int(np.ceil(ideal_step * 2.0)))
    dp = np.full((m, n), 1e9, dtype=np.float64)
    parent = np.full((m, n), -1, dtype=np.int32)

    for i in range(min_start, max_start + 1):
        start_cost = 0.0
        if prev_last_c is not None:
            # Measure actual pixel change on 128x72 gradient + 64x36 color across the cut
            cut_c = float(np.mean(np.abs(sub_c[i] - prev_last_c))) * 255.0
            if i == 1 and max_start >= 2 and len(step_diffs) > 0 and step_diffs[0] < 2.6:
                # Frame 0->1 is an on-twos hold; prefer starting at frame 2 where the new drawing begins
                start_cost += 0.04
            elif i == 2 and len(step_diffs) > 0 and step_diffs[0] >= 3.2:
                # Frame 1 already has a new drawing; do not skip frame 1
                start_cost += 0.05 * (step_diffs[0] / max(1.0, median_step))
            if cut_c > 18.0:
                start_cost += 0.55 * (cut_c / 255.0)
        dp[0, i] = start_cost

    for t in range(1, m):
        rem = m - 1 - t
        i_min = min_start + t
        i_max = n - 1 - rem
        for i in range(i_min, i_max + 1):
            p_min = max(min_start + t - 1, i - max_step)
            p_max = i - 1
            best_val = 1e9
            best_p = -1
            arc_err = 0.08 * abs(float(cum_mot[i]) - float(alphas[t]))
            sharp_bonus = -0.002 * float(sub_s[i])
            for p in range(p_min, p_max + 1):
                step = i - p
                step_cost = 0.008 * ((step - ideal_step) ** 2)
                # When skipping frame(s) p+1..i-1, penalize by how much drawing change occurred in the skipped step(s):
                # skipping an on-twos duplicate hold (step_diffs[k] ~ 1.2) is nearly free;
                # skipping a real sakuga drawing (step_diffs[k] > 5.0) is heavily penalized.
                skip_drawing_cost = 0.0
                if step > 1:
                    for k_skip in range(p, i - 1):
                        hold_diff = float(step_diffs[k_skip])
                        skip_drawing_cost += 0.025 * ((hold_diff / max(1.5, median_step)) ** 2)
                frame_delta = float(np.mean(np.abs(sub_c[i] - sub_c[p]))) * 255.0
                stall_cost = 0.030 * max(0.0, 2.0 - frame_delta) / 2.0 if frame_delta < 2.0 else 0.0
                val = (
                    dp[t - 1, p]
                    + arc_err
                    + step_cost
                    + skip_drawing_cost
                    + stall_cost
                    + sharp_bonus
                )
                if val < best_val:
                    best_val = val
                    best_p = p
            dp[t, i] = best_val
            parent[t, i] = best_p

    # Always lock the final frame to n - 1 (frame 38) when it converges to last_kf_c so the target keyframe is reached
    min_end = max(min_start + m - 1, n - 2)
    end_costs = dp[m - 1, min_end:n].copy()
    for idx_e, e_pos in enumerate(range(min_end, n)):
        end_anchor_diff = float(np.mean(np.abs(sub_c[e_pos] - last_kf_c)))
        not_last_pen = 0.03 * m if e_pos < n - 1 and end_anchor_diff > float(np.mean(np.abs(sub_c[n - 1] - last_kf_c))) else 0.0
        end_costs[idx_e] += 0.50 * end_anchor_diff * m + not_last_pen
    best_end = min_end + int(np.argmin(end_costs))
    total_cost = float(end_costs[best_end - min_end])

    sched_local = [0] * m
    cur = best_end
    for t in range(m - 1, -1, -1):
        sched_local[t] = int(cur)
        cur = int(parent[t, cur])

    sched_global = [lo + idx for idx in sched_local]
    sel_c = cand_c[sched_global]
    active_mot = float(np.mean(np.abs(sel_c[1:] - sel_c[:-1]))) * 255.0 if m > 1 else 0.0
    out_anchor_mae = float(np.mean(np.abs(sel_c[-1] - last_kf_c))) * 255.0
    return sched_global, total_cost / m, active_mot, out_anchor_mae


def collect_pool(batches: list[tuple[str, str]]) -> dict[str, list[dict]]:
    by_shot: dict[str, list[dict]] = {}
    for round_name, batch_rel in batches:
        batch_dir = EXP / batch_rel
        metrics_path = batch_dir / "studio_batch_metrics.json"
        manifest_path = EXP / "outputs" / "rounds" / round_name / "manifest.json"
        if not metrics_path.exists() or not manifest_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        shot_meta = {s["shot_id"]: s for s in manifest["storyboards"]}
        for job in metrics["jobs"]:
            mp4 = batch_dir / job["mp4_file"]
            if not mp4.exists():
                continue
            s_info = shot_meta[job["shot_id"]]
            mode = "FL2VA" if s_info.get("last_frame") else "I2VA"
            by_shot.setdefault(job["shot_id"], []).append(
                {
                    "round": round_name,
                    "batch": batch_rel,
                    "job_id": f"{round_name}_{job['job_id']}" if round_name != "round01a" and job["job_id"] in {x["job_id"] for x in by_shot.get(job["shot_id"], [])} else job["job_id"],
                    "raw_job_id": job["job_id"],
                    "shot_id": job["shot_id"],
                    "seed": int(job["seed"]),
                    "mode": mode,
                    "num_frames": int(s_info["num_frames"]),
                    "mp4": mp4,
                    "prod_quality_score": float(job.get("prod_quality_score", 80.0)),
                    "S_fl2v_pixel": float(job.get("S_fl2v_pixel", 90.0)),
                }
            )
    return by_shot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyboard", type=Path, default=EXP / "code" / "sub04_storyboard_20shots.json")
    parser.add_argument("--write-edls", action="store_true")
    args = parser.parse_args()

    sb = json.loads(args.storyboard.read_text())
    slots = sb["slots"]
    batches = [(b["round"], b["batch_dir"]) for b in sb["batches"]]
    by_shot = collect_pool(batches)

    # Pre-cache decoded features per mp4
    feature_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get_features(mp4_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = str(mp4_path)
        if key not in feature_cache:
            frames = decode(mp4_path)
            feature_cache[key] = extract_clip_features(frames)
        return feature_cache[key]

    def build_edl_parts(
        strategy: str,
        scale: float = 1.0,
    ) -> tuple[list[dict], list[dict]]:
        parts, diagnostics = [], []
        prev_last_c = None
        for idx, slot in enumerate(slots):
            slot_id = slot["slot_id"]
            is_last = idx == len(slots) - 1
            base_target = int(slot["target_frames_30s"])
            target_frames = max(12, round(base_target * scale))
            first_c, _ = load_kf_feature(slot["first"])
            last_c, _ = load_kf_feature(slot["last"])

            # Gather candidates for this slot
            cands: list[tuple[dict, tuple[int, int] | None]] = []
            for c in by_shot.get(slot_id, []):
                cands.append((c, None))
            if strategy in ("hybrid", "dp_best"):
                for alt in slot.get("ablation_sources", []):
                    alt_id = alt["shot_id"]
                    sub_r = (int(alt["sub_range"][0]), int(alt["sub_range"][1])) if "sub_range" in alt else None
                    for c in by_shot.get(alt_id, []):
                        cands.append((c, sub_r))

            if not cands:
                raise RuntimeError(f"No completed candidates found for slot {slot_id}")

            if strategy == "raw_seed42":
                chosen_cand, _ = next(
                    (x for x in cands if x[0]["mode"] == "FL2VA" and x[0]["num_frames"] == 39 and x[0]["seed"] == 42),
                    cands[0],
                )
                c_c, _, _ = get_features(chosen_cand["mp4"])
                sched = np.linspace(0, chosen_cand["num_frames"] - 1, target_frames).round().astype(int).tolist()
                anchor_seam = (
                    float(np.mean(np.abs(c_c[0] - prev_last_c))) * 255.0
                    if prev_last_c is not None
                    else 0.0
                )
                cut_mot = (
                    float(np.mean(np.abs(c_c[sched[0]] - prev_last_c))) * 255.0
                    if prev_last_c is not None
                    else 0.0
                )
                sel_c = c_c[sched]
                act_mot = float(np.mean(np.abs(sel_c[1:] - sel_c[:-1]))) * 255.0
                out_mae = float(np.mean(np.abs(sel_c[-1] - last_c))) * 255.0
                prev_last_c = c_c[sched[-1]]
                best_entry = {
                    "cand": chosen_cand,
                    "schedule": sched,
                    "seam_mae_255": round(anchor_seam, 2),
                    "cut_motion_mae_255": round(cut_mot, 2),
                    "active_motion_255": round(act_mot, 2),
                    "out_anchor_mae_255": round(out_mae, 2),
                    "combined_cost": 0.0,
                }
                all_scores = [best_entry]
            else:
                best_entry = None
                all_scores = []
                for cand, sub_r in cands:
                    if strategy == "hybrid" and slot.get("ablation_sources"):
                        ab_ids = {a["shot_id"] for a in slot["ablation_sources"]}
                        if cand["shot_id"] not in ab_ids:
                            continue
                    c_c, c_g, c_s = get_features(cand["mp4"])
                    lo_idx = sub_r[0] if sub_r is not None else 0
                    sched, dp_cost, act_mot, out_mae = solve_velocity_regularized_schedule(
                        c_c,
                        c_g,
                        c_s,
                        first_c,
                        last_c,
                        target_frames,
                        prev_last_c=prev_last_c,
                        sub_range=sub_r,
                    )
                    anchor_seam = (
                        float(np.mean(np.abs(c_c[lo_idx] - prev_last_c))) * 255.0
                        if prev_last_c is not None
                        else 0.0
                    )
                    cut_mot = (
                        float(np.mean(np.abs(c_c[sched[0]] - prev_last_c))) * 255.0
                        if prev_last_c is not None
                        else 0.0
                    )
                    combined = (
                        dp_cost * 100.0
                        + 0.40 * anchor_seam
                        + 0.30 * out_mae
                        - 0.14 * act_mot
                        - 0.06 * cand["prod_quality_score"]
                    )
                    entry = {
                        "cand": cand,
                        "schedule": sched,
                        "seam_mae_255": round(anchor_seam, 2),
                        "cut_motion_mae_255": round(cut_mot, 2),
                        "active_motion_255": round(act_mot, 2),
                        "out_anchor_mae_255": round(out_mae, 2),
                        "combined_cost": round(combined, 3),
                        "last_c": c_c[sched[-1]],
                    }
                    all_scores.append(entry)
                    if best_entry is None or entry["combined_cost"] < best_entry["combined_cost"]:
                        best_entry = entry

                assert best_entry is not None
                prev_last_c = best_entry["last_c"]

            chosen = best_entry["cand"]
            sched = best_entry["schedule"]
            part = {
                "slot": slot_id,
                "mode": f"{chosen['mode']}_{chosen['num_frames']}f",
                "batch": chosen["batch"],
                "job_id": chosen["raw_job_id"],
                "start_frame": sched[0],
                "end_frame_exclusive": sched[-1] + 1,
                "selected_frames": sched,
                "target_frames": target_frames,
                "keyframes": f"{slot['first']}->{slot['last']}",
                "reason": (
                    f"{slot['title']} [{chosen['mode']} {chosen['num_frames']}f seed={chosen['seed']}]："
                    f"选帧 {sched[0]}..{sched[-1]} -> {target_frames}f "
                    f"(anchor_seam={best_entry['seam_mae_255']}, cut_flow={best_entry['cut_motion_mae_255']}, "
                    f"mot={best_entry['active_motion_255']}, out_anchor={best_entry['out_anchor_mae_255']})"
                ),
            }
            if idx == 0:
                part["fade_in_frames"] = 4
            if is_last:
                part["fade_out_frames"] = 8
            parts.append(part)
            diagnostics.append(
                {
                    "slot_id": slot_id,
                    "title": slot["title"],
                    "keyframes": f"{slot['first']}->{slot['last']}",
                    "target_frames": target_frames,
                    "selected_job": chosen["job_id"],
                    "selected_mode": f"{chosen['mode']}_{chosen['num_frames']}f",
                    "selected_seed": chosen["seed"],
                    "schedule_span": f"{sched[0]}..{sched[-1]} ({len(sched)}f)",
                    "seam_mae_255": best_entry["seam_mae_255"],
                    "cut_motion_mae_255": best_entry["cut_motion_mae_255"],
                    "active_motion_255": best_entry["active_motion_255"],
                    "out_anchor_mae_255": best_entry["out_anchor_mae_255"],
                    "candidates": [
                        {
                            "job_id": e["cand"]["job_id"],
                            "mode": f"{e['cand']['mode']}_{e['cand']['num_frames']}f",
                            "seed": e["cand"]["seed"],
                            "schedule_span": f"{e['schedule'][0]}..{e['schedule'][-1]}",
                            "seam_mae_255": e["seam_mae_255"],
                            "cut_motion_mae_255": e["cut_motion_mae_255"],
                            "active_motion_255": e["active_motion_255"],
                            "out_anchor_mae_255": e["out_anchor_mae_255"],
                            "prod_quality_score": e["cand"]["prod_quality_score"],
                            "combined_cost": e["combined_cost"],
                        }
                        for e in all_scores
                    ],
                }
            )
        return parts, diagnostics

    raw_parts, raw_diag = build_edl_parts("raw_seed42", scale=1.0)
    hyb_parts, hyb_diag = build_edl_parts("hybrid", scale=1.0)
    dp_parts, dp_diag = build_edl_parts("dp_best", scale=1.0)
    tight_parts, _ = build_edl_parts("dp_best", scale=0.867)

    edls = {
        "edl_workflow_a_raw.json": {
            "name": "workflow_a_fl2va39_raw",
            "target_total_frames": sum(p["target_frames"] for p in raw_parts),
            "description": "工作流A基线（30.00s / 720帧）：固定Seed42 + 39帧FL2VA均匀抽帧（0..38），未剔除切镜处的双端重复锚点帧（导致19处切镜瞬间出现2帧静止顿挫）。",
            "parts": raw_parts,
        },
        "edl_workflow_b_hybrid.json": {
            "name": "workflow_b_hybrid_i2va_73f",
            "target_total_frames": sum(p["target_frames"] for p in hyb_parts),
            "description": "工作流B混合消融版（30.00s / 720帧）：在蓄势/觉醒/燃焰/散景分镜使用I2VA（仅首帧）并在连段分镜使用73帧（3.04s）跨拍长窗口，暴露尾帧失锁带来的剧烈切镜跳变。",
            "parts": hyb_parts,
        },
        "edl_director_30s00.json": {
            "name": "director_master_30s00",
            "target_total_frames": sum(p["target_frames"] for p in dp_parts),
            "description": "最终30秒导演母带（30.00s / 720帧）：20镜同源关键帧链 + 多Seed抽卡 + 弧长速度正则化单调动态规划（Arc-Length Velocity-Regularized DP），剔除切镜重复帧（1..38）并保留高能作画段。",
            "parts": dp_parts,
        },
        "edl_director_tight.json": {
            "name": "director_master_tight",
            "target_total_frames": sum(p["target_frames"] for p in tight_parts),
            "description": f"节奏强化导演版（{sum(p['target_frames'] for p in tight_parts) / 24:.2f}s / {sum(p['target_frames'] for p in tight_parts)}帧）：在20镜多Seed最优链基础上抽取每镜最高速动作核（0.867×时长压缩）。",
            "parts": tight_parts,
        },
    }

    if args.write_edls:
        for fname, content in edls.items():
            (EXP / "code" / fname).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")

    def summarize(diag: list[dict]) -> dict:
        seams = [d["seam_mae_255"] for d in diag[1:]]
        cuts = [d["cut_motion_mae_255"] for d in diag[1:]]
        mots = [d["active_motion_255"] for d in diag]
        outs = [d["out_anchor_mae_255"] for d in diag]
        freezes = sum(1 for c in cuts if c < 2.2)
        jumps = sum(1 for s in seams if s > 12.0)
        return {
            "mean_anchor_seam_mae_255": round(float(np.mean(seams)), 2) if seams else 0.0,
            "max_anchor_seam_mae_255": round(float(np.max(seams)), 2) if seams else 0.0,
            "mean_cut_motion_mae_255": round(float(np.mean(cuts)), 2) if cuts else 0.0,
            "boundary_freeze_count": freezes,
            "jump_cut_count": jumps,
            "mean_active_motion_255": round(float(np.mean(mots)), 2),
            "mean_out_anchor_mae_255": round(float(np.mean(outs)), 2),
        }

    report = {
        "workflow_comparison": {
            "workflow_a_fl2va39_raw": summarize(raw_diag),
            "workflow_b_hybrid_i2va_73f": summarize(hyb_diag),
            "director_master_30s00": summarize(dp_diag),
        },
        "director_30s00_slots": dp_diag,
        "hybrid_ablation_slots": hyb_diag,
    }
    out_report = EXP / "outputs" / "review" / "gacha_and_dp_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "OK", "comparison": report["workflow_comparison"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
