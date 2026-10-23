#!/usr/bin/env python3
"""
Experiment 03 Main Orchestrator:
MiniMax H3-Base-FL2VA Sol-Attn & Community Acceleration Pareto Ablation
on Google Colab NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB GDDR7, SM 12.0, --gpu G4).
"""

import argparse, concurrent.futures, datetime, gc, importlib.metadata, json, logging, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

from ablation_config import (
    A100_BASELINE_CU, A100_BASELINE_E2E_SEC, COMFY_ROOT, CUSTOM_SOL_DIR, FallbackAlertDetector, G4_ABLATION_MATRIX,
    G4_CU_RATE_PER_HOUR, G4_CU_RATE_PER_SEC, OFFICIAL_PROMPT, OUTPUT_DIR, assert_expected_kernel_calls,
    bootstrap_and_download_parallel, clone_av_latent, clone_conditioning, prepare_first_and_last_frames,
)
from eval_metrics import QualityEvaluator, compute_pareto_frontier, save_video_with_audio
from microbench_verifications import run_microbenchmark_and_verifications

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def reconcile_session_ledger(metrics_json_path: Path, ledger_excerpt_path: Path) -> None:
    """Merges sanitized Colab CLI session ledger telemetry into g4_sol_attn_ablation_metrics.json."""
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    with open(ledger_excerpt_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    events, derived = {ev["event"]: ev for ev in ledger.get("events", [])}, ledger.get("derived_accounting", {})
    lt = payload["lifecycle_telemetry"]
    pre_bal = float(events["pre_session_ccu_info"]["response_body"]["currentBalance"])
    post_bal = float(events["post_session_immediate_ccu_info"]["response_body"]["currentBalance"])
    session_wall_sec = float(derived["assign_to_unassign_wall_sec"])
    session_theo_cu, settled_deducted_cu = round(session_wall_sec * G4_CU_RATE_PER_SEC, 4), round(pre_bal - post_bal, 4)
    core_e2e_sec, script_wall_sec = float(lt["core_12runs_e2e_sum_sec"]), float(lt["script_total_wall_sec"])
    t_assign = datetime.datetime.fromisoformat(events["vm_assign_post"]["local_timestamp"])
    t_kernel = datetime.datetime.fromisoformat(events["kernel_websocket_ready"]["local_timestamp"])
    assign_to_kernel_sec, outside_script_sec = round((t_kernel - t_assign).total_seconds(), 3), round(session_wall_sec - script_wall_sec, 3)
    known_phases_sec = sum(
        float(lt[k]) for k in (
            "phase1_total_sec", "cond_encode_sec", "dit_load_sec", "microbench_and_verifications_sec",
            "lightweight_1step_warmup_sec", "ablation_loop_wall_sec", "eval_4d_sec",
        )
    )
    for old_key in ("session_post_script_export_tar_download_stop_sec", "async_unsettled_tail_cu"):
        lt.pop(old_key, None)
    lt.update({
        "script_unallocated_interphase_sec": round(script_wall_sec - known_phases_sec, 3),
        "session_outside_script_timer_sec": outside_script_sec,
        "session_assign_to_kernel_ready_sec": assign_to_kernel_sec,
        "session_post_kernel_ready_outside_script_sec": round(outside_script_sec - assign_to_kernel_sec, 3),
        "session_assign_local_ts": events["vm_assign_post"]["local_timestamp"],
        "session_kernel_ready_local_ts": events["kernel_websocket_ready"]["local_timestamp"],
        "session_unassign_local_ts": events["vm_unassign_post"]["local_timestamp"],
        "session_total_wall_sec": round(session_wall_sec, 3),
        "session_theoretical_cu": session_theo_cu,
        "session_core_inference_share_pct": round(100.0 * core_e2e_sec / session_wall_sec, 2),
        "bundle_download_sec": round(float(events["tar_bundle_download_complete_and_unassign_get"]["download_elapsed_sec"]), 3),
        "tar_bundle_bytes": int(events["tar_bundle_download_start"]["content_length_bytes"]),
        "pre_session_balance_cu": pre_bal, "post_session_balance_cu": post_bal,
        "post_session_verified_settled_ts": events["post_session_settled_verification_ccu_info"]["local_timestamp"],
        "post_session_verified_balance_cu": float(events["post_session_settled_verification_ccu_info"]["response_body"]["currentBalance"]),
        "settled_ledger_deducted_cu": settled_deducted_cu,
        "theoretical_minus_ledger_diff_cu": round(session_theo_cu - settled_deducted_cu, 4),
        "ledger_excerpt_file": f"outputs/{ledger_excerpt_path.name}",
    })
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Reconciled session ledger telemetry from {ledger_excerpt_path.name} into {metrics_json_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 03 G4 Sol-Attn Pareto Ablation Orchestrator")
    parser.add_argument("--reconcile-ledger-only", action="store_true", help="Reconcile local colab_session_ledger_excerpt.json into metrics JSON.")
    args = parser.parse_args()

    if args.reconcile_ledger_only:
        _script_dir = Path(__file__).resolve().parent
        _exp_root = _script_dir.parent if _script_dir.name == "code" else _script_dir
        here_out = _exp_root / "outputs"
        reconcile_session_ledger(here_out / "g4_sol_attn_ablation_metrics.json", here_out / "colab_session_ledger_excerpt.json")
        return

    t_script_start = time.perf_counter()
    phase1_stats = bootstrap_and_download_parallel()

    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(COMFY_ROOT / "custom_nodes"))

    import comfy_kitchen as ck
    import numpy as np
    import psutil
    import torch
    import comfy.ldm.minimax.model as minimax_model_mod
    import comfy.ldm.modules.attention as comfy_attn_mod
    import comfy.model_management as mm
    import comfy.ops
    import nodes
    from comfy_extras.nodes_audio import VAEDecodeAudio
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
    from comfy_extras.nodes_sparse_attention import BlockSparseAttention
    import comfyui_sol_attn.minimax as community_sol_mod
    from comfyui_sol_attn.minimax import (
        MiniMaxH3FusedModulation, MiniMaxH3MemoryEfficientSolAttentionPatch,
        MiniMaxH3ScheduledSolAttentionPatch, _TauSchedule,
    )
    from sageattention import sageattn

    comfy_attn_mod.sageattn = sageattn
    comfy_attn_mod.SAGE_ATTENTION_IS_AVAILABLE = True

    props = torch.cuda.get_device_properties(0)
    print("\n" + "=" * 96)
    print(f"🖥️ [Phase 2/5] Single-Pass Conditioning & Resident Model Loading on {props.name} (SM {props.major}.{props.minor})")
    print("=" * 96)

    detector = FallbackAlertDetector()
    logging.getLogger().addHandler(detector)

    width, height, length, steps, seed = 1344, 768, 124, 8, 42
    first_tensor, last_tensor, first_np, last_np = prepare_first_and_last_frames(width, height)

    t_enc0 = time.perf_counter()
    clip = nodes.CLIPLoader().load_clip("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax")[0]
    vae_loader = nodes.VAELoader()
    video_vae = vae_loader.load_vae("minimax_h3_video_vae_int8_convrot.safetensors")[0]
    audio_vae = vae_loader.load_vae("minimax_h3_audio_vae_fp32.safetensors")[0]

    with torch.inference_mode():
        base_cond, base_latent = MiniMaxH3ImageToVideo.execute(
            clip=clip, vae=video_vae, prompt=OFFICIAL_PROMPT,
            width=width, height=height, length=length, first_frame=first_tensor, last_frame=last_tensor,
        )
    torch.cuda.synchronize()
    cond_encode_sec = time.perf_counter() - t_enc0
    print(f"  ✅ Shared Text + First/Last Keyframe Conditioning encoded in {cond_encode_sec:.2f}s")

    del clip
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()

    t_load0 = time.perf_counter()
    base_model = nodes.UNETLoader().load_unet("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default")[0]
    lora_model = nodes.LoraLoaderModelOnly().load_lora_model_only(
        base_model, "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 1.0
    )[0]
    mm.load_models_gpu([lora_model])
    torch.cuda.synchronize()
    dit_load_sec = time.perf_counter() - t_load0
    print(f"  ✅ FL2VA DiT + 8-Step Turbo LoRA loaded and resident in G4 VRAM in {dit_load_sec:.2f}s")

    t_micro0 = time.perf_counter()
    microbench_results = run_microbenchmark_and_verifications(
        comfy_attn_mod, ck, sageattn, community_sol_mod, lora_model, detector
    )
    microbench_sec = time.perf_counter() - t_micro0

    kernel_counters = {
        "sdpa": 0, "sage": 0, "ck_int8": 0,
        "ck_sol_chunked": 0, "ck_sol_direct": 0, "triton_sol": 0, "fused_mod_blocks": 0,
    }
    def wrap_counter(key: str, fn):
        def _wrapped(*args, **kwargs):
            kernel_counters[key] += 1
            return fn(*args, **kwargs)
        return _wrapped

    comfy.ops.scaled_dot_product_attention = wrap_counter("sdpa", comfy.ops.scaled_dot_product_attention)
    comfy_attn_mod.sageattn = wrap_counter("sage", comfy_attn_mod.sageattn)
    ck.int8_attention = wrap_counter("ck_int8", ck.int8_attention)
    ck.int8_attention_from_prequantized = wrap_counter("ck_int8", ck.int8_attention_from_prequantized)
    ck.sol_attn_chunked = wrap_counter("ck_sol_chunked", ck.sol_attn_chunked)
    ck.sol_attn = wrap_counter("ck_sol_direct", ck.sol_attn)
    community_sol_mod.sol_attn = wrap_counter("triton_sol", community_sol_mod.sol_attn)
    community_sol_mod._FusionLog.hit = wrap_counter("fused_mod_blocks", community_sol_mod._FusionLog.hit)

    def set_dense_attention_backend(backend: str) -> None:
        fn_map = {
            "sdpa": comfy_attn_mod.attention_pytorch,
            "sage": comfy_attn_mod.attention_sage,
            "ck_int8": comfy_attn_mod.attention_comfy_kitchen_int8,
        }
        if backend not in fn_map:
            raise ValueError(f"Unknown dense_backend: {backend}")
        comfy_attn_mod.optimized_attention = fn_map[backend]
        minimax_model_mod.optimized_attention = fn_map[backend]

    sampler_obj = KSamplerSelect.execute("res_multistep")[0]
    vae_decoder = nodes.VAEDecode()

    print("\n" + "=" * 96)
    print("🔥 [Phase 3C/5] Lightweight 1-Step DiT + VAE Warmup (Replacing Redundant 2x8-Step Warmup)")
    print("=" * 96)
    t_warm0 = time.perf_counter()
    set_dense_attention_backend("sage")
    warm_model = lora_model.clone()
    warm_cond = clone_conditioning(base_cond)
    warm_latent = clone_av_latent(base_latent)
    warm_noise = RandomNoise.execute(seed)[0]
    with torch.inference_mode():
        warm_sigmas = BasicScheduler.execute(warm_model, "simple", 1, 1.0)[0]
        warm_guider = BasicGuider.execute(warm_model, warm_cond)[0]
        _, warm_denoised = SamplerCustomAdvanced.execute(warm_noise, warm_guider, sampler_obj, warm_sigmas, warm_latent)
        _ = vae_decoder.decode(video_vae, warm_denoised)[0]
        _ = VAEDecodeAudio.execute(audio_vae, warm_denoised)[0]
    torch.cuda.synchronize()
    del warm_model, warm_cond, warm_latent, warm_guider, warm_denoised
    gc.collect()
    torch.cuda.empty_cache()
    warmup_sec = time.perf_counter() - t_warm0
    print(f"  ✅ 1-Step DiT + Video/Audio VAE Warmup finished in {warmup_sec:.2f}s (saved {154.47 - warmup_sec:.2f}s vs Exp 02)")

    print("\n" + "=" * 96)
    print("🚀 [Phase 4/5] Executing 12-Run Sol-Attn & Community Fusion Pareto Ablation Matrix on G4")
    print("=" * 96)

    orig_tau_progress = _TauSchedule.progress

    def fixed_flow_shift_progress(self, sigma):
        if sigma is None:
            return 1.0
        sig = max(min(float(sigma), 1.0), 0.0)
        t_unshifted = sig / (12.0 - 11.0 * sig)
        return min(max(1.0 - t_unshifted, 0.0), 1.0)

    def build_patched_model(cfg: Dict[str, Any]) -> Any:
        set_dense_attention_backend(cfg["dense_backend"])
        work_model = lora_model.clone()
        sparse_mode = cfg["sparse_mode"]
        _TauSchedule.progress = orig_tau_progress

        if sparse_mode in ("ck_chunked", "ck_direct"):
            work_model = BlockSparseAttention.execute(
                work_model,
                selection={"selection": "sol-attn", "tau": float(cfg["tau"])},
                start_percent=float(cfg["start_percent"]),
                end_percent=float(cfg["end_percent"]),
                dense_blocks=str(cfg.get("dense_blocks", "")),
                min_tokens=12288,
                extra_tokens=int(cfg.get("extra_tokens", 0)),
                sink_conditioning=str(cfg.get("sink_conditioning", "exact_kv_and_rows")),
                verbose=False,
            )[0]
            if sparse_mode == "ck_direct":
                work_model.model_options["transformer_options"].get("patches_replace", {}).pop("dit", None)
        elif sparse_mode == "triton_mem_eff":
            work_model = MiniMaxH3MemoryEfficientSolAttentionPatch().patch(
                model=work_model, enabled=True, tau=float(cfg["tau"]), min_tokens=4096, strict=True,
                thresh_type="diag", int8_qk=bool(cfg.get("int8_qk", True)), int8_pv=bool(cfg.get("int8_pv", False)),
                sink_conditioning=str(cfg.get("sink_conditioning", "exact_kv_and_rows")),
                dense_blocks=str(cfg.get("dense_blocks", "")),
            )[0]
        elif sparse_mode == "triton_scheduled_fixed":
            _TauSchedule.progress = fixed_flow_shift_progress
            work_model = MiniMaxH3ScheduledSolAttentionPatch().patch(
                model=work_model, enabled=True, tau_start=float(cfg["tau_start"]), tau_end=float(cfg["tau_end"]),
                curve=str(cfg["curve"]), min_tokens=4096, strict=True, dense_percent=float(cfg["start_percent"]),
                thresh_type="diag", int8_qk=bool(cfg.get("int8_qk", True)), int8_pv=bool(cfg.get("int8_pv", True)),
                sink_conditioning=str(cfg.get("sink_conditioning", "exact_kv_and_rows")),
                dense_blocks=str(cfg.get("dense_blocks", "")),
            )[0]
        elif sparse_mode != "none":
            raise ValueError(f"Unknown sparse_mode: {sparse_mode}")

        if cfg.get("fused_mod", False):
            work_model = MiniMaxH3FusedModulation().patch(model=work_model, enabled=True)[0]
        return work_model

    raw_run_outputs: Dict[int, Dict[str, Any]] = {}
    all_frames_dict: Dict[int, Any] = {}
    mp4_futures: Dict[int, concurrent.futures.Future] = {}
    mp4_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    t_ablation_loop0 = time.perf_counter()
    for cfg in G4_ABLATION_MATRIX:
        run_id, code, label = cfg["run_id"], cfg["code"], cfg["label"]
        print(f"\n▶️ Run {run_id:02d}/12 {code} — {label}\n   {cfg['desc']}")

        work_model = build_patched_model(cfg)
        cond_copy = clone_conditioning(base_cond)
        latent_copy = clone_av_latent(base_latent)
        noise_obj = RandomNoise.execute(seed)[0]

        detector.fallback_errors.clear()
        for k in kernel_counters:
            kernel_counters[k] = 0

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        with torch.inference_mode():
            sigmas = BasicScheduler.execute(work_model, "simple", steps, 1.0)[0]
            guider = BasicGuider.execute(work_model, cond_copy)[0]
            out_latent, out_denoised = SamplerCustomAdvanced.execute(noise_obj, guider, sampler_obj, sigmas, latent_copy)
            torch.cuda.synchronize()
            t_denoise = time.perf_counter() - t0
            dit_kernel_calls = dict(kernel_counters)

            t_v0 = time.perf_counter()
            video_frames_t = vae_decoder.decode(video_vae, out_denoised)[0]
            torch.cuda.synchronize()
            t_vae_video = time.perf_counter() - t_v0

            t_a0 = time.perf_counter()
            audio_dict = VAEDecodeAudio.execute(audio_vae, out_denoised)[0]
            torch.cuda.synchronize()
            t_vae_audio = time.perf_counter() - t_a0

        t_e2e = time.perf_counter() - t0
        peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

        if detector.fallback_errors:
            raise RuntimeError(f"Fallback detected during {label}: {detector.fallback_errors}")
        assert_expected_kernel_calls(cfg, dit_kernel_calls)

        frames_np = video_frames_t.detach().cpu().float().numpy()
        if frames_np.ndim == 5:
            frames_np = frames_np[0]
        waveform_np = audio_dict["waveform"].detach().cpu().float().numpy()
        if waveform_np.ndim == 3:
            waveform_np = waveform_np[0]
        sample_rate = int(audio_dict["sample_rate"])

        assert np.all(np.isfinite(frames_np)) and np.all(np.isfinite(waveform_np))
        del guider, out_latent, out_denoised, video_frames_t, audio_dict, work_model
        gc.collect()

        mp4_path = OUTPUT_DIR / f"g4_run{run_id:02d}_{label}.mp4"
        mp4_futures[run_id] = mp4_pool.submit(save_video_with_audio, frames_np, waveform_np, sample_rate, mp4_path, 24)

        cu_denoise = t_denoise * G4_CU_RATE_PER_SEC
        cu_e2e = t_e2e * G4_CU_RATE_PER_SEC
        total_attn_calls = sum(
            dit_kernel_calls[k] for k in ("sdpa", "sage", "ck_int8", "ck_sol_chunked", "ck_sol_direct", "triton_sol")
        )
        print(
            f"   ✅ Denoise: {t_denoise:.2f}s ({cu_denoise:.4f} CU) | VideoVAE: {t_vae_video:.2f}s | "
            f"AudioVAE: {t_vae_audio:.2f}s | E2E: {t_e2e:.2f}s ({cu_e2e:.4f} CU) | PeakVRAM: {peak_vram_gib:.2f} GiB\n"
            f"   📊 DiT Calls (total={total_attn_calls}): {dit_kernel_calls}"
        )

        all_frames_dict[run_id] = frames_np
        raw_run_outputs[run_id] = {
            **cfg,
            "_raw_denoise_sec": float(t_denoise), "_raw_e2e_sec": float(t_e2e),
            "denoise_sec": round(float(t_denoise), 4), "vae_video_sec": round(float(t_vae_video), 4),
            "vae_audio_sec": round(float(t_vae_audio), 4), "e2e_sec": round(float(t_e2e), 4),
            "cu_denoise": round(float(cu_denoise), 6), "cu_e2e": round(float(cu_e2e), 6),
            "videos_per_100_cu": round(100.0 / cu_e2e, 2), "peak_vram_gib": round(float(peak_vram_gib), 3),
            "kernel_calls": dit_kernel_calls, "total_dit_attn_calls": int(total_attn_calls),
            "mp4_file": mp4_path.name, "waveform_np": waveform_np, "sample_rate": sample_rate,
        }

    ablation_loop_wall_sec = time.perf_counter() - t_ablation_loop0
    sum_core_e2e_sec = sum(r["_raw_e2e_sec"] for r in raw_run_outputs.values())
    sum_core_denoise_sec = sum(r["_raw_denoise_sec"] for r in raw_run_outputs.values())

    _TauSchedule.progress = orig_tau_progress
    del lora_model, base_model, video_vae, audio_vae, base_cond, base_latent
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print("📊 [Phase 5/5] Computing 4-Dimensional Quality Scores & Pareto Frontier on G4")
    print("=" * 96)

    t_eval0 = time.perf_counter()
    evaluator = QualityEvaluator(device="cuda")
    gt_frames, gt_audio = all_frames_dict[1], raw_run_outputs[1]["waveform_np"]
    t_base_e2e, t_base_denoise = raw_run_outputs[1]["_raw_e2e_sec"], raw_run_outputs[1]["_raw_denoise_sec"]
    t_sage_e2e, t_sage_denoise = raw_run_outputs[2]["_raw_e2e_sec"], raw_run_outputs[2]["_raw_denoise_sec"]

    baseline_av_corr = None
    final_results: List[Dict[str, Any]] = []

    for cfg in G4_ABLATION_MATRIX:
        run_id = cfg["run_id"]
        r = raw_run_outputs[run_id]
        frames_t, audio_t, sr_t = all_frames_dict[run_id], r.pop("waveform_np"), r.pop("sample_rate")
        is_base = run_id == 1

        ref_m = evaluator.compute_reference_metrics(frames_t, gt_frames, is_baseline=is_base)
        vbench_m, fl2v_m, motion_energy_t = evaluator.compute_vbench_and_fl2v_metrics(frames_t, first_np, last_np)
        audio_m, baseline_av_corr = evaluator.compute_audio_and_avsync_metrics(
            audio_t, gt_audio, motion_energy_t, baseline_av_corr=baseline_av_corr, is_baseline=is_base, sample_rate=sr_t
        )

        q_total = 0.35 * ref_m["S_ref"] + 0.25 * vbench_m["S_vbench"] + 0.25 * fl2v_m["S_fl2v"] + 0.15 * audio_m["S_audio"]
        q_total_linear_supp = (
            0.35 * ref_m["S_ref"] + 0.25 * vbench_m["S_vbench"] + 0.25 * fl2v_m["S_fl2v"] + 0.15 * audio_m["S_audio_linear_supp"]
        )

        raw_denoise, raw_e2e = r["_raw_denoise_sec"], r["_raw_e2e_sec"]
        r["denoise_speedup_vs_sdpa"] = round(float(t_base_denoise / raw_denoise), 4)
        r["e2e_speedup_vs_sdpa"] = round(float(t_base_e2e / raw_e2e), 4)
        r["denoise_speedup_vs_sage"] = round(float(t_sage_denoise / raw_denoise), 4)
        r["e2e_speedup_vs_sage"] = round(float(t_sage_e2e / raw_e2e), 4)
        r["e2e_speedup_vs_a100_base"] = round(float(A100_BASELINE_E2E_SEC / raw_e2e), 4)
        r["cu_saving_pct_vs_g4_sdpa"] = round(100.0 * (1.0 - raw_e2e / t_base_e2e), 2)
        r["cu_saving_pct_vs_g4_sage"] = round(100.0 * (1.0 - raw_e2e / t_sage_e2e), 2)
        r["cu_saving_pct_vs_a100_base"] = round(100.0 * (1.0 - (raw_e2e * G4_CU_RATE_PER_SEC) / A100_BASELINE_CU), 2)

        r.update({
            "_raw_Q_total": float(q_total), "_raw_Q_total_linear_supp": float(q_total_linear_supp), "_raw_S_fl2v": float(fl2v_m["S_fl2v"]),
            "S_ref": round(float(ref_m["S_ref"]), 4), "S_vbench": round(float(vbench_m["S_vbench"]), 4),
            "S_fl2v": round(float(fl2v_m["S_fl2v"]), 4), "S_audio": round(float(audio_m["S_audio"]), 4),
            "S_audio_linear_supp": round(float(audio_m["S_audio_linear_supp"]), 4),
            "Q_total": round(float(q_total), 4), "Q_total_linear_supp": round(float(q_total_linear_supp), 4),
            "ref_metrics": ref_m, "vbench_metrics": vbench_m, "fl2v_metrics": fl2v_m, "audio_metrics": audio_m,
        })
        final_results.append(r)

    evaluator.close()

    mp4_encode_secs = {run_id: round(fut.result(), 3) for run_id, fut in mp4_futures.items()}
    mp4_pool.shutdown(wait=True)
    for r in final_results:
        r["async_mp4_encode_sec"] = mp4_encode_secs.get(r["run_id"], 0.0)

    pareto_e2e_ids = compute_pareto_frontier(final_results, time_key="_raw_e2e_sec", quality_key="_raw_Q_total")
    pareto_denoise_ids = compute_pareto_frontier(final_results, time_key="_raw_denoise_sec", quality_key="_raw_Q_total")
    pareto_e2e_lin_ids = compute_pareto_frontier(final_results, time_key="_raw_e2e_sec", quality_key="_raw_Q_total_linear_supp")
    pareto_denoise_lin_ids = compute_pareto_frontier(final_results, time_key="_raw_denoise_sec", quality_key="_raw_Q_total_linear_supp")
    pareto_fl2v_ids = compute_pareto_frontier(final_results, time_key="_raw_e2e_sec", quality_key="_raw_S_fl2v")

    for r in final_results:
        for k in ("_raw_denoise_sec", "_raw_e2e_sec", "_raw_Q_total", "_raw_Q_total_linear_supp", "_raw_S_fl2v"):
            r.pop(k, None)
        r["is_pareto_e2e_qtotal"] = r["run_id"] in pareto_e2e_ids
        r["is_pareto_denoise_qtotal"] = r["run_id"] in pareto_denoise_ids
        r["is_pareto_e2e_qtotal_linear_supp"] = r["run_id"] in pareto_e2e_lin_ids
        r["is_pareto_e2e_fl2v"] = r["run_id"] in pareto_fl2v_ids

    eval_sec = time.perf_counter() - t_eval0
    script_total_sec = time.perf_counter() - t_script_start
    core_inference_share_of_script_pct = 100.0 * sum_core_e2e_sec / script_total_sec

    comfy_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(COMFY_ROOT), text=True).strip()
    community_sol_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(CUSTOM_SOL_DIR), text=True).strip()

    payload = {
        "experiment": "03-g4-sol-attn-pareto-ablation",
        "hardware": {
            "gpu_name": props.name, "compute_capability": f"{props.major}.{props.minor}",
            "vram_gib": round(props.total_memory / (1024 ** 3), 2), "ram_gib": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "cpu_count": os.cpu_count(), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "triton_version": importlib.metadata.version("triton"), "sageattention_version": importlib.metadata.version("sageattention"),
            "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
            "comfyui_commit": comfy_commit, "community_sol_attn_commit": community_sol_commit,
        },
        "config": {
            "resolution": f"{width}x{height}", "frames": length, "fps": 24, "steps": steps, "seed": seed,
            "joint_sequence_tokens": 42058, "dit_blocks": 50, "attention_heads": 56, "head_dim": 128,
            "g4_cu_rate_per_hour": G4_CU_RATE_PER_HOUR,
        },
        "lifecycle_telemetry": {
            **phase1_stats,
            "cond_encode_sec": round(cond_encode_sec, 3), "dit_load_sec": round(dit_load_sec, 3),
            "microbench_and_verifications_sec": round(microbench_sec, 3), "lightweight_1step_warmup_sec": round(warmup_sec, 3),
            "core_12runs_e2e_sum_sec": round(sum_core_e2e_sec, 3), "core_12runs_denoise_sum_sec": round(sum_core_denoise_sec, 3),
            "ablation_loop_wall_sec": round(ablation_loop_wall_sec, 3),
            "async_mp4_cpu_sum_sec_overlapped": round(sum(mp4_encode_secs.values()), 3),
            "eval_4d_sec": round(eval_sec, 3), "script_total_wall_sec": round(script_total_sec, 3),
            "core_inference_cu": round(sum_core_e2e_sec * G4_CU_RATE_PER_SEC, 4),
            "script_total_cu": round(script_total_sec * G4_CU_RATE_PER_SEC, 4),
            "core_inference_share_of_script_pct": round(core_inference_share_of_script_pct, 2),
        },
        "pareto_frontiers": {
            "e2e_sec_vs_Q_total": pareto_e2e_ids, "denoise_sec_vs_Q_total": pareto_denoise_ids,
            "e2e_sec_vs_Q_total_linear_supp": pareto_e2e_lin_ids, "denoise_sec_vs_Q_total_linear_supp": pareto_denoise_lin_ids,
            "e2e_sec_vs_S_fl2v": pareto_fl2v_ids,
        },
        "microbenchmark": microbench_results,
        "runs": final_results,
    }

    json_path = OUTPUT_DIR / "g4_sol_attn_ablation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    bundle_tar = Path("/content/exp03_g4_outputs_bundle.tar")
    subprocess.run(["tar", "-cf", str(bundle_tar), "-C", str(OUTPUT_DIR), "."], check=True)
    print(f"✅ Experiment 03 Completed in {script_total_sec:.1f}s! Bundle saved to {bundle_tar}")


if __name__ == "__main__":
    main()
