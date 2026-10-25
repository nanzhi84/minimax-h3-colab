#!/usr/bin/env python3
"""
Experiment 07 Remote G4 Orchestrator (`experiments/07-bf16-dual-route-2x2-ablation/code/remote_exp07_engine.py`).
Executes the 2x2 BF16 Dual-Route & Dual-Tier Ablation Matrix on Google Colab Blackwell G4 (--gpu G4),
runs the 4D QualityEvaluator + Production Metrics, and uploads `exp07_outputs_bundle.tar` to Google Drive
via ADR-04 Stage 1 (datacenter-to-datacenter REST API) for immediate 0-waste G4 shutdown.
"""

import argparse, concurrent.futures, gc, hashlib, importlib.metadata, json, logging, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

if "/content" not in sys.path:
    sys.path.insert(0, "/content")

from exp07_config import (
    COMFY_ROOT, EXP03_INT8_RUN12_8STEP_E2E_SEC, EXP03_INT8_SDPA_8STEP_E2E_SEC, EXP07_2X2_MATRIX,
    FallbackAlertDetector, G4_CU_RATE_PER_HOUR, G4_CU_RATE_PER_SEC, OFFICIAL_FL2VA_PROMPT,
    OFFICIAL_REF2VA_PROMPT, OUTPUT_DIR, assert_expected_kernel_calls, bootstrap_and_download_bf16_parallel,
    clone_av_latent, clone_conditioning, prepare_first_and_last_frames,
)
from eval_metrics import QualityEvaluator, save_video_with_audio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

DEFAULT_ACCESS_TOKEN = os.environ.get("EXP07_ACCESS_TOKEN", "__EXP07_ACCESS_TOKEN__")
DEFAULT_QUOTA_PROJECT = os.environ.get("EXP07_QUOTA_PROJECT", "__EXP07_QUOTA_PROJECT__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 07 Remote G4 2x2 BF16 Ablation Engine")
    parser.add_argument("--access-token", default=DEFAULT_ACCESS_TOKEN)
    parser.add_argument("--quota-project", default=DEFAULT_QUOTA_PROJECT)
    parser.add_argument("--output-tar", type=Path, default=Path("/content/exp07_outputs_bundle.tar"))
    args, _ = parser.parse_known_args()

    t_script_start = time.perf_counter()
    phase1_stats = bootstrap_and_download_bf16_parallel()

    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(COMFY_ROOT / "custom_nodes"))

    import comfy_kitchen as ck, psutil, requests, torch
    import comfy.ldm.minimax.model as minimax_model_mod
    import comfy.ldm.modules.attention as comfy_attn_mod
    import comfy.model_management as mm
    import comfy.ops, nodes
    from comfy_extras.nodes_audio import VAEDecodeAudio
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
    from comfy_extras.nodes_sparse_attention import BlockSparseAttention
    import comfyui_sol_attn.minimax as community_sol_mod
    from comfyui_sol_attn.minimax import MiniMaxH3FusedModulation
    from sageattention import sageattn

    props = torch.cuda.get_device_properties(0)
    print("\n" + "=" * 96)
    print(f"🧠 [Phase 2/5] Dual-Route BF16 Conditioning Pre-Encode (Qwen3-VL-32B BF16 + Video VAE FP16) on {props.name}")
    print("=" * 96)

    detector = FallbackAlertDetector()
    logging.getLogger().addHandler(detector)

    width, height, length, seed = 1344, 768, 124, 42
    first_tensor, last_tensor, first_np, last_np = prepare_first_and_last_frames(width, height)

    t_p2_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    t_vlm_load0 = time.perf_counter()
    clip = nodes.CLIPLoader().load_clip("qwen3vl_32b_minimax_h3_bf16.safetensors", type="minimax")[0]
    vae_loader = nodes.VAELoader()
    video_vae = vae_loader.load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
    audio_vae = vae_loader.load_vae("minimax_h3_audio_vae_fp32.safetensors")[0]
    vlm_vae_load_sec = time.perf_counter() - t_vlm_load0

    t_fl_enc0 = time.perf_counter()
    with torch.inference_mode():
        fl2va_cond, fl2va_latent = MiniMaxH3ImageToVideo.execute(
            clip=clip, vae=video_vae, prompt=OFFICIAL_FL2VA_PROMPT,
            width=width, height=height, length=length, first_frame=first_tensor, last_frame=last_tensor,
        )
    torch.cuda.synchronize()
    fl2va_cond_encode_sec = time.perf_counter() - t_fl_enc0
    print(f"  ✅ Encoded FL2VA conditioning (first_frame + last_frame) in {fl2va_cond_encode_sec:.2f}s")

    t_ref_enc0 = time.perf_counter()
    with torch.inference_mode():
        ref2va_cond, ref2va_latent = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=OFFICIAL_REF2VA_PROMPT, width=width, height=height, length=length,
            ref_image_size="match", vae=video_vae, audio_vae=audio_vae,
            ref_images={"ref_image_0": first_tensor, "ref_image_1": last_tensor},
        )
    torch.cuda.synchronize()
    ref2va_cond_encode_sec = time.perf_counter() - t_ref_enc0
    print(f"  ✅ Encoded Ref2VA conditioning (<Picture 1> + <Picture 2>, ref_image_size='match') in {ref2va_cond_encode_sec:.2f}s")

    vram_before_vlm_unload_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    peak_p2_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    t_un0 = time.perf_counter()
    del clip
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    vlm_unload_sec = time.perf_counter() - t_un0
    vram_after_vlm_unload_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    phase2_total_sec = time.perf_counter() - t_p2_start
    print(
        f"  ✅ Phase 2 completed in {phase2_total_sec:.2f}s | Peak VLM VRAM: {peak_p2_vram_gib:.2f} GiB | "
        f"Freed Qwen3-VL BF16 VRAM: {vram_before_vlm_unload_gib - vram_after_vlm_unload_gib:.2f} GiB (Now: {vram_after_vlm_unload_gib:.2f} GiB)"
    )

    route_cond_map = {"fl2va": (fl2va_cond, fl2va_latent), "ref2va": (ref2va_cond, ref2va_latent)}
    kernel_counters = {"sdpa": 0, "sage": 0, "ck_int8": 0, "ck_sol_chunked": 0, "ck_sol_direct": 0, "triton_sol": 0, "fused_mod_blocks": 0}

    def _wrap_counter(key: str, fn):
        def _wrapped(*a, **kw):
            kernel_counters[key] += 1
            return fn(*a, **kw)
        return _wrapped

    comfy_attn_mod.sageattn = _wrap_counter("sage", sageattn)
    comfy_attn_mod.SAGE_ATTENTION_IS_AVAILABLE = True
    comfy.ops.scaled_dot_product_attention = _wrap_counter("sdpa", comfy.ops.scaled_dot_product_attention)
    ck.int8_attention = _wrap_counter("ck_int8", ck.int8_attention)
    ck.int8_attention_from_prequantized = _wrap_counter("ck_int8", ck.int8_attention_from_prequantized)
    ck.sol_attn_chunked = _wrap_counter("ck_sol_chunked", ck.sol_attn_chunked)
    ck.sol_attn = _wrap_counter("ck_sol_direct", ck.sol_attn)
    community_sol_mod.sol_attn = _wrap_counter("triton_sol", community_sol_mod.sol_attn)
    community_sol_mod._FusionLog.hit = _wrap_counter("fused_mod_blocks", community_sol_mod._FusionLog.hit)

    sampler_obj = KSamplerSelect.execute("res_multistep")[0]
    vae_decoder = nodes.VAEDecode()
    mp4_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    mp4_futures: List[concurrent.futures.Future] = []
    cached_outputs: Dict[str, Dict[str, Any]] = {}
    run_metrics_list: List[Dict[str, Any]] = []

    print("\n" + "=" * 96)
    print("🚀 [Phase 3/5] Executing 2x2 BF16 Dual-Route Ablation Matrix (FL2VA Bucket -> Ref2VA Bucket)")
    print("=" * 96)

    t_ablation_loop0 = time.perf_counter()
    total_dit_load_sec, total_warmup_sec = 0.0, 0.0

    for route_key, unet_filename in [
        ("fl2va", "minimax_h3_fl2va_pruned_bf16.safetensors"),
        ("ref2va", "minimax_h3_ref2va_pruned_bf16.safetensors"),
    ]:
        route_runs = [cfg for cfg in EXP07_2X2_MATRIX if cfg["model_route"] == route_key]
        print("\n" + "-" * 96)
        print(f"📦 Loading Route Backbone: {unet_filename} (route={route_key}, {len(route_runs)} profiles)")
        print("-" * 96)

        t_d0 = time.perf_counter()
        base_unet = nodes.UNETLoader().load_unet(unet_filename, "default")[0]
        mm.load_models_gpu([base_unet])
        torch.cuda.synchronize()
        d_sec = time.perf_counter() - t_d0
        total_dit_load_sec += d_sec
        print(f"  ✅ Loaded {unet_filename} to GPU in {d_sec:.2f}s (Allocated VRAM: {torch.cuda.memory_allocated()/(1024**3):.2f} GiB)")

        t_w0 = time.perf_counter()
        b_cond, b_lat = route_cond_map[route_key]
        for warm_mode in ("sdpa", "sol"):
            w_model = base_unet.clone()
            if warm_mode == "sdpa":
                comfy_attn_mod.optimized_attention = comfy_attn_mod.attention_pytorch
                minimax_model_mod.optimized_attention = comfy_attn_mod.attention_pytorch
            else:
                comfy_attn_mod.optimized_attention = comfy_attn_mod.attention_sage
                minimax_model_mod.optimized_attention = comfy_attn_mod.attention_sage
                w_model = BlockSparseAttention.execute(
                    w_model, selection={"selection": "sol-attn", "tau": 1.3}, start_percent=0.0, end_percent=1.0,
                    dense_blocks="", min_tokens=4096, extra_tokens=256, sink_conditioning="exact_kv_and_rows", verbose=False,
                )[0]
                w_model.model_options["transformer_options"].get("patches_replace", {}).pop("dit", None)
            w_model = MiniMaxH3FusedModulation().patch(model=w_model, enabled=True)[0]
            w_cond, w_lat = clone_conditioning(b_cond), clone_av_latent(b_lat)
            with torch.inference_mode():
                w_sig = BasicScheduler.execute(w_model, "simple", 1, 1.0)[0]
                w_gui = BasicGuider.execute(w_model, w_cond)[0]
                _, w_den = SamplerCustomAdvanced.execute(RandomNoise.execute(seed)[0], w_gui, sampler_obj, w_sig, w_lat)
                if route_key == "fl2va" and warm_mode == "sdpa":
                    _ = vae_decoder.decode(video_vae, w_den)[0]
                    _ = VAEDecodeAudio.execute(audio_vae, w_den)[0]
            torch.cuda.synchronize()
            del w_model, w_cond, w_lat, w_gui, w_den
        gc.collect()
        torch.cuda.empty_cache()
        w_sec = time.perf_counter() - t_w0
        total_warmup_sec += w_sec
        print(f"  🔥 Route '{route_key}' 1-step kernel warmup finished in {w_sec:.2f}s")

        for cfg in route_runs:
            run_id, steps, sched_name = cfg["id"], int(cfg["steps"]), str(cfg["scheduler"])
            print(f"\n▶️ [{cfg['run_idx']:02d}/04] {cfg['short']} {run_id} ({cfg['notes']})")

            t_patch0 = time.perf_counter()
            if cfg["lora_file"]:
                work_model = nodes.LoraLoaderModelOnly().load_lora_model_only(base_unet.clone(), cfg["lora_file"], 1.0)[0]
                mm.load_models_gpu([work_model])
            else:
                work_model = base_unet.clone()

            if cfg["dense_backend"] == "sdpa":
                comfy_attn_mod.optimized_attention = comfy_attn_mod.attention_pytorch
                minimax_model_mod.optimized_attention = comfy_attn_mod.attention_pytorch
            elif cfg["dense_backend"] == "sage":
                comfy_attn_mod.optimized_attention = comfy_attn_mod.attention_sage
                minimax_model_mod.optimized_attention = comfy_attn_mod.attention_sage

            if cfg["sparse_mode"] == "ck_sol_direct":
                work_model = BlockSparseAttention.execute(
                    work_model, selection={"selection": "sol-attn", "tau": float(cfg["tau"])},
                    start_percent=float(cfg["start_percent"]), end_percent=float(cfg["end_percent"]),
                    dense_blocks=str(cfg["dense_blocks"]), min_tokens=4096, extra_tokens=int(cfg["extra_tokens"]),
                    sink_conditioning=str(cfg["sink_conditioning"]), verbose=False,
                )[0]
                work_model.model_options["transformer_options"].get("patches_replace", {}).pop("dit", None)

            if cfg["fused_mod"]:
                work_model = MiniMaxH3FusedModulation().patch(model=work_model, enabled=True)[0]

            cond_copy, latent_copy = clone_conditioning(b_cond), clone_av_latent(b_lat)
            noise_obj = RandomNoise.execute(seed)[0]
            detector.fallback_errors.clear()
            for k in kernel_counters:
                kernel_counters[k] = 0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            patch_prep_sec = time.perf_counter() - t_patch0

            t_r0 = time.perf_counter()
            with torch.inference_mode():
                sigmas = BasicScheduler.execute(work_model, sched_name, steps, 1.0)[0]
                guider = BasicGuider.execute(work_model, cond_copy)[0]
                _, out_denoised = SamplerCustomAdvanced.execute(noise_obj, guider, sampler_obj, sigmas, latent_copy)
                torch.cuda.synchronize()
                t_denoise = time.perf_counter() - t_r0
                dit_calls = dict(kernel_counters)

                t_v0 = time.perf_counter()
                video_frames_t = vae_decoder.decode(video_vae, out_denoised)[0]
                torch.cuda.synchronize()
                t_vae_video = time.perf_counter() - t_v0

                t_a0 = time.perf_counter()
                audio_dict = VAEDecodeAudio.execute(audio_vae, out_denoised)[0]
                torch.cuda.synchronize()
                t_vae_audio = time.perf_counter() - t_a0

            t_e2e = time.perf_counter() - t_r0
            peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            alloc_vram_gib = torch.cuda.memory_allocated() / (1024 ** 3)

            if detector.fallback_errors:
                raise RuntimeError(f"Silent fallback detected during {run_id}: {detector.fallback_errors}")
            assert_expected_kernel_calls(cfg, dit_calls)

            frames_np = video_frames_t.detach().cpu().float().numpy()
            if frames_np.ndim == 5:
                frames_np = frames_np[0]
            waveform_np = audio_dict["waveform"].detach().cpu().float().numpy()
            if waveform_np.ndim == 3:
                waveform_np = waveform_np[0]
            sample_rate = int(audio_dict["sample_rate"])

            mp4_path = OUTPUT_DIR / f"{run_id}.mp4"
            mp4_futures.append(mp4_pool.submit(save_video_with_audio, frames_np, waveform_np, sample_rate, mp4_path, 24))
            cached_outputs[run_id] = {"frames": frames_np, "waveform": waveform_np, "sr": sample_rate}

            cu_denoise, cu_e2e = t_denoise * G4_CU_RATE_PER_SEC, t_e2e * G4_CU_RATE_PER_SEC
            print(
                f"   ✅ Denoise ({steps} steps, {sched_name}): {t_denoise:.2f}s ({t_denoise/steps:.2f}s/step, {cu_denoise:.4f} CU) | "
                f"VideoVAE: {t_vae_video:.2f}s | AudioVAE: {t_vae_audio:.2f}s | E2E: {t_e2e:.2f}s ({cu_e2e:.4f} CU) | "
                f"PeakVRAM: {peak_vram_gib:.2f} GiB | Calls: {dit_calls}"
            )

            run_metrics_list.append({
                "index": cfg["run_idx"], "id": run_id, "short": cfg["short"], "profile_id": cfg["profile_id"],
                "model_route": cfg["model_route"], "profile": cfg["profile"], "unet_file": cfg["unet_file"],
                "lora_file": cfg["lora_file"], "steps": steps, "scheduler": sched_name, "notes": cfg["notes"],
                "config": cfg, "patch_prep_sec": round(patch_prep_sec, 3), "denoise_sec": round(t_denoise, 4),
                "sec_per_step": round(t_denoise / steps, 4), "vae_video_sec": round(t_vae_video, 4),
                "vae_audio_sec": round(t_vae_audio, 4), "vae_total_sec": round(t_vae_video + t_vae_audio, 4),
                "e2e_sec": round(t_e2e, 4), "cu_denoise": round(cu_denoise, 5), "cu_e2e": round(cu_e2e, 5),
                "peak_vram_gib": round(peak_vram_gib, 3), "allocated_vram_gib": round(alloc_vram_gib, 3),
                "kernel_calls": dit_calls, "video_file": f"outputs/{mp4_path.name}",
            })

            del work_model, guider, out_denoised, video_frames_t, audio_dict, cond_copy, latent_copy
            gc.collect()
            torch.cuda.empty_cache()

        del base_unet
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  🧹 Unloaded {unet_filename} (VRAM now: {torch.cuda.memory_allocated()/(1024**3):.2f} GiB)")

    ablation_loop_wall_sec = time.perf_counter() - t_ablation_loop0
    for fut in mp4_futures:
        fut.result()
    mp4_pool.shutdown(wait=True)

    del video_vae, audio_vae, fl2va_cond, fl2va_latent, ref2va_cond, ref2va_latent
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print("📐 [Phase 4/5] 4-Dimensional Orthogonal Quality Evaluation + Production Metrics across 4 Runs")
    print("=" * 96)
    t_eval0 = time.perf_counter()
    evaluator = QualityEvaluator(device="cuda")

    run01_id = "Run01_FL2VA_BF16_Quality_20Step_SDPA"
    run01_out = cached_outputs[run01_id]
    by_id = {m["id"]: m for m in run_metrics_list}
    for m in run_metrics_list:
        rid = m["id"]
        route_base_id = m["config"]["route_baseline_id"]
        route_base_m, route_base_out = by_id[route_base_id], cached_outputs[route_base_id]

        m["speedup_vs_route_quality_denoise"] = round(route_base_m["denoise_sec"] / m["denoise_sec"], 4)
        m["speedup_vs_route_quality_e2e"] = round(route_base_m["e2e_sec"] / m["e2e_sec"], 4)
        m["speedup_vs_run01_denoise"] = round(by_id[run01_id]["denoise_sec"] / m["denoise_sec"], 4)
        m["speedup_vs_run01_e2e"] = round(by_id[run01_id]["e2e_sec"] / m["e2e_sec"], 4)
        m["speedup_vs_exp03_int8_sdpa_8step"] = round(EXP03_INT8_SDPA_8STEP_E2E_SEC / m["e2e_sec"], 4)
        m["speedup_vs_exp03_int8_run12_8step"] = round(EXP03_INT8_RUN12_8STEP_E2E_SEC / m["e2e_sec"], 4)

        cur = cached_outputs[rid]
        q_res = evaluator.evaluate_run_dual_ref(
            frames=cur["frames"], waveform=cur["waveform"], sr=cur["sr"],
            route_ref_frames=route_base_out["frames"], route_ref_waveform=route_base_out["waveform"],
            run01_ref_frames=run01_out["frames"], run01_ref_waveform=run01_out["waveform"],
            first_ref=first_np, last_ref=last_np,
            is_route_baseline=(rid == route_base_id), is_run01_baseline=(rid == run01_id),
        )
        m["quality"] = q_res
        print(
            f"  📊 {m['short']} ({m['profile_id']}): "
            f"Q_total(intra-route)={q_res['Q_total']:.2f} (LinearSupp={q_res['Q_total_linear_supp']:.2f}) | "
            f"Q_total(vs Run01)={q_res['Q_total_vs_run01']:.2f} | "
            f"S_ref={q_res['ref_fidelity']['S_ref']:.2f}, S_vbench={q_res['vbench_temporal']['S_vbench']:.2f}, "
            f"S_fl2v={q_res['fl2v_alignment']['S_fl2v']:.2f}, S_audio={q_res['audio_quality']['S_audio']:.2f} | "
            f"prod_quality={q_res['production_metrics']['prod_quality_score']:.2f} (S_fl2v_pixel={q_res['production_metrics']['S_fl2v_pixel']:.2f})"
        )

    eval_total_sec = time.perf_counter() - t_eval0
    script_total_sec = time.perf_counter() - t_script_start
    core_4runs_e2e_sec = sum(m["e2e_sec"] for m in run_metrics_list)
    core_4runs_denoise_sec = sum(m["denoise_sec"] for m in run_metrics_list)

    metrics_payload = {
        "experiment": "07-bf16-dual-route-2x2-ablation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "gpu_name": props.name, "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_gib": round(props.total_memory / (1024 ** 3), 2),
            "system_ram_gib": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "cpu_count_logical": os.cpu_count(), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "triton_version": importlib.metadata.version("triton"),
            "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
            "sageattention_version": importlib.metadata.version("sageattention"),
            "comfyui_commit": "f5ea88af1c46c50a39792d00af80e5af941f99f6",
            "comfyui_sol_attn_commit": "69d2e4009313e5cc3a1a70ab092ed6bb54fd5cd3",
            "g4_cu_rate_per_hour": G4_CU_RATE_PER_HOUR,
        },
        "generation_params": {
            "width": width, "height": height, "length": length, "fps": 24, "seed": seed,
            "sampler": "res_multistep", "cfg": 1.0,
            "fl2va_prompt": OFFICIAL_FL2VA_PROMPT, "ref2va_prompt": OFFICIAL_REF2VA_PROMPT,
        },
        "lifecycle_telemetry": {
            **phase1_stats, "vlm_vae_load_sec": round(vlm_vae_load_sec, 3),
            "fl2va_cond_encode_sec": round(fl2va_cond_encode_sec, 3),
            "ref2va_cond_encode_sec": round(ref2va_cond_encode_sec, 3),
            "vlm_unload_sec": round(vlm_unload_sec, 3), "phase2_total_sec": round(phase2_total_sec, 3),
            "peak_phase2_vlm_vram_gib": round(peak_p2_vram_gib, 3),
            "freed_vlm_vram_gib": round(vram_before_vlm_unload_gib - vram_after_vlm_unload_gib, 3),
            "total_dit_load_sec": round(total_dit_load_sec, 3), "total_warmup_sec": round(total_warmup_sec, 3),
            "ablation_loop_wall_sec": round(ablation_loop_wall_sec, 3),
            "core_4runs_denoise_sum_sec": round(core_4runs_denoise_sec, 3),
            "core_4runs_e2e_sum_sec": round(core_4runs_e2e_sec, 3),
            "eval_4d_sec": round(eval_total_sec, 3), "script_total_wall_sec": round(script_total_sec, 3),
            "core_4runs_cu": round(core_4runs_e2e_sec * G4_CU_RATE_PER_SEC, 4),
            "script_total_cu": round(script_total_sec * G4_CU_RATE_PER_SEC, 4),
            "core_inference_cu_share_pct": round(100.0 * core_4runs_e2e_sec / script_total_sec, 2),
        },
        "runs": run_metrics_list,
    }

    metrics_json_str = json.dumps(metrics_payload, indent=2, ensure_ascii=False)
    (OUTPUT_DIR / "exp07_bf16_2x2_metrics.json").write_text(metrics_json_str, encoding="utf-8")
    import base64
    print("EXP07_METRICS_JSON_B64:" + base64.b64encode(metrics_json_str.encode("utf-8")).decode("ascii"))

    t_tar0 = time.perf_counter()
    subprocess.run(["tar", "-cf", str(args.output_tar), "-C", str(OUTPUT_DIR), "."], check=True)
    tar_pack_sec = time.perf_counter() - t_tar0
    tar_data = args.output_tar.read_bytes()
    sz, md5, sha256 = len(tar_data), hashlib.md5(tar_data).hexdigest(), hashlib.sha256(tar_data).hexdigest()

    t_up0 = time.perf_counter()
    for attempt in range(4):
        try:
            r = requests.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=media&fields=id,name,size,md5Checksum",
                headers={
                    "Authorization": f"Bearer {args.access_token}", "X-Goog-User-Project": args.quota_project,
                    "Content-Type": "application/x-tar", "Content-Length": str(sz),
                },
                data=tar_data, timeout=180,
            )
            r.raise_for_status()
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.8 * (attempt + 1))
    stage1_sec = time.perf_counter() - t_up0
    info = r.json()
    assert int(info["size"]) == sz and info.get("md5Checksum") == md5
    stage1_mb_s = (sz / 1e6) / max(stage1_sec, 1e-6)
    print(f"\n⚡ [ADR-04 Stage 1 Complete] Uploaded {sz/1e6:.2f} MB to GDrive in {stage1_sec:.3f}s ({stage1_mb_s:.1f} MB/s)")
    print("EXP07_STAGE1_EGRESS_JSON:" + json.dumps({"file_id": info["id"], "raw_bytes": sz, "md5": md5, "sha256": sha256, "tar_pack_sec": round(tar_pack_sec, 4), "stage1_sec": round(stage1_sec, 4), "stage1_mb_s": round(stage1_mb_s, 2)}))


if __name__ == "__main__":
    main()
