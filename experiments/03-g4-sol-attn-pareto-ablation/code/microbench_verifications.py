#!/usr/bin/env python3
"""
Operator-Level Microbenchmark, Synthetic Workspace Block-Routing Sparsity Telemetry,
Discovery 1 (Chunked 2x-Pass Bootstrap vs Direct Zero-Copy Sol-Attn),
Triton Segmented FusedModulation Bit-Exactness Verification,
and Discovery 2 (Flow-Matching Shift=12.0 Schedule Audit) for Experiment 03.
"""

import gc
import time
from typing import Any, Dict, Tuple

from ablation_config import FallbackAlertDetector


def measure_ck_sol_sparsity(
    q_nhd: Any,
    k_nhd: Any,
    v_nhd: Any,
    tau: float,
    token_aug: int = 0,
    sink_blocks: Tuple[int, int] = (0, 75),
    sink_q: Tuple[int, int] = (67, 75),
) -> Dict[str, Any]:
    """
    Executes comfy_kitchen's C++ _C.sol_attn on an explicit workspace to read back
    routed key-block counts (`cnt`, NTB=658) and extra outlier token counts (`tokCnt`).
    Note: comfy-kitchen v0.2.40 (sol_layout.cuh:46) defines TOK_GROUP = 2, so each head
    has ng = (nq + 1) // 2 = 329 groups (18,424 total tokCnt entries across BH=56 heads).
    """
    import torch
    from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack

    batch, t, h, d = q_nhd.shape
    scale = d ** -0.5
    p = _C.sol_attn_plan(batch, t, h, token_aug=int(token_aug))
    ws = torch.empty(p["total"], dtype=torch.uint8, device=q_nhd.device)
    out = torch.empty_like(q_nhd)
    _C.sol_attn(
        _wrap_for_dlpack(q_nhd),
        _wrap_for_dlpack(k_nhd),
        _wrap_for_dlpack(v_nhd),
        _wrap_for_dlpack(out),
        _wrap_for_dlpack(ws),
        batch,
        t,
        h,
        d,
        float(tau),
        float(scale),
        int(sink_blocks[0]),
        int(sink_blocks[1]),
        int(sink_q[0]),
        int(sink_q[1]),
        torch.cuda.current_stream(q_nhd.device).cuda_stream,
        key_bias=None,
        threshold=None,
        block_len=None,
        tail=True,
        token_aug=int(token_aug),
    )
    torch.cuda.synchronize()
    bh = batch * h
    ntb = int(p["NTB"])
    nq = int(p["NQ"])
    cnt_view = ws[p["cnt"] : p["cnt"] + bh * nq * 4].view(torch.int32).view(bh, nq).float()
    video_q_cnt = cnt_view[:, sink_blocks[1] :]
    mean_routed_all = float(cnt_view.mean().item())
    mean_routed_video_q = float(video_q_cnt.mean().item())
    pct_routed_all = 100.0 * mean_routed_all / ntb
    pct_routed_video_q = 100.0 * mean_routed_video_q / ntb

    mean_extra_tokens_routed = 0.0
    if int(token_aug) > 0 and p.get("tokCnt", 0) > 0:
        # TOK_GROUP = 2 in comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:46
        ng = (nq + 1) // 2
        tok_cnt_view = ws[p["tokCnt"] : p["tokCnt"] + bh * ng * 4].view(torch.int32).view(bh, ng).float()
        mean_extra_tokens_routed = round(float(tok_cnt_view.mean().item()), 2)

    return {
        "tau": tau,
        "token_aug": token_aug,
        "total_blocks_ntb": ntb,
        "workspace_mib": round(p["total"] / (1024 ** 2), 2),
        "mean_routed_blocks_all_q": round(mean_routed_all, 2),
        "mean_routed_blocks_video_q": round(mean_routed_video_q, 2),
        "mean_extra_tokens_routed_per_group": mean_extra_tokens_routed,
        "kept_exact_pct_all_q": round(pct_routed_all, 2),
        "kept_exact_pct_video_q": round(pct_routed_video_q, 2),
        "sparsity_pct_video_q": round(100.0 - pct_routed_video_q, 2),
    }


def run_microbenchmark_and_verifications(
    comfy_attn_mod: Any,
    ck_mod: Any,
    sageattn_fn: Any,
    comfyui_sol_mod: Any,
    lora_model: Any,
    detector: FallbackAlertDetector,
) -> Dict[str, Any]:
    """
    Phase 3A & 3B:
      1. Operator-Level Microbenchmark at exact MiniMaxH3Model Joint AV Attention shape:
         B=1, H=56, N=42058, D=128, dtype=bfloat16 (50.717194436608 TFLOPs/call, 400 DiT calls/video).
      2. Synthetic Block-Routing Sparsity Telemetry from comfy-kitchen _C.sol_attn_plan workspace.
      3. Discovery 1 Verification: Single H3 DiTBlock benchmark comparing 4K-chunked producer
         (1st-step double-pass qkv_proj bootstrap vs warm step) against Direct Zero-Copy ck.sol_attn,
         and bit-exactness verification of Triton MiniMaxH3FusedModulation vs Eager AdaLN/Gate.
      4. Discovery 2 Verification: Stock _TauSchedule vs Flow-Shift-Corrected (shift=12.0) _TauSchedule
         across the 8 sigmas of MiniMaxH3's simple scheduler.
    """
    import torch
    import torch.nn.functional as F
    from comfy.ldm.modules.attention import AttentionTensorContainer

    print("\n" + "=" * 96)
    print("🔬 [Phase 3A/5] Operator & Block Microbenchmark at Exact H3 Shape (B=1, H=56, N=42058, D=128)")
    print("=" * 96)

    B, H, N, D = 1, 56, 42058, 128
    flops = 4.0 * B * H * (N ** 2) * D  # 50.717194436608 TFLOPs per call
    gen = torch.Generator(device="cuda").manual_seed(42)

    q_hnd = torch.randn((B, H, N, D), generator=gen, device="cuda", dtype=torch.bfloat16)
    k_hnd = torch.randn((B, H, N, D), generator=gen, device="cuda", dtype=torch.bfloat16)
    v_hnd = torch.randn((B, H, N, D), generator=gen, device="cuda", dtype=torch.bfloat16)

    qkv_fused = torch.randn((N, 3 * H * D), generator=gen, device="cuda", dtype=torch.bfloat16)
    qf, kf, vf = qkv_fused.split(H * D, dim=-1)
    q_nhd_strided = qf.view(1, N, H, D)
    k_nhd_strided = kf.view(1, N, H, D)
    v_nhd_strided = vf.view(1, N, H, D)
    q_hnd_strided = q_nhd_strided.transpose(1, 2)
    k_hnd_strided = k_nhd_strided.transpose(1, 2)
    v_hnd_strided = v_nhd_strided.transpose(1, 2)

    with torch.inference_mode():
        out_ref_sdpa = F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd, is_causal=False)
        out_ref_flat = out_ref_sdpa.transpose(1, 2).reshape(1, -1).float()
        out_ref_strided = F.scaled_dot_product_attention(
            q_hnd_strided, k_hnd_strided, v_hnd_strided, is_causal=False
        ).transpose(1, 2).reshape(1, -1).float()

    def fidelity_vs_ref(out_tensor: Any, ref_flat: Any) -> Tuple[float, float]:
        cand_flat = out_tensor.reshape(1, -1).float()
        cos_sim = float(F.cosine_similarity(ref_flat, cand_flat).item())
        rel_l1 = float((torch.abs(ref_flat - cand_flat).mean() / (torch.abs(ref_flat).mean() + 1e-8)).item())
        return round(cos_sim, 6), round(rel_l1, 6)

    def bench_op(
        name: str,
        fn,
        ref_flat: Any = None,
        warmup: int = 3,
        reps: int = 8,
    ) -> Dict[str, Any]:
        detector.fallback_errors.clear()
        out = None
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn()
        torch.cuda.synchronize()
        if detector.fallback_errors:
            raise RuntimeError(f"Fallback detected during microbenchmark {name}: {detector.fallback_errors}")
        avg_sec = (time.perf_counter() - t0) / reps
        avg_ms = avg_sec * 1000.0
        eff_tflops = round((flops / avg_sec) / 1e12, 2)
        est_400_calls_sec = round(avg_sec * 400.0, 2)
        cos_sim, rel_l1 = (None, None)
        if ref_flat is not None and out is not None:
            cos_sim, rel_l1 = fidelity_vs_ref(out, ref_flat)
        fid_str = f"Cos={cos_sim:.6f}, RelL1={rel_l1:.4f}" if cos_sim is not None else "N/A"
        print(
            f"  [{name:<42}] {avg_ms:7.2f} ms/call | {eff_tflops:6.1f} eff-TFLOPS | "
            f"400 calls={est_400_calls_sec:5.2f}s | {fid_str}"
        )
        return {
            "avg_ms": round(avg_ms, 3),
            "effective_tflops": eff_tflops,
            "est_400_calls_sec": est_400_calls_sec,
            "cosine_sim_vs_sdpa": cos_sim,
            "rel_l1_vs_sdpa": rel_l1,
        }

    triton_sol_fn = comfyui_sol_mod.sol_attn
    ops_bench: Dict[str, Any] = {}
    with torch.inference_mode():
        q_c, k_c, v_c = AttentionTensorContainer(q_hnd_strided), AttentionTensorContainer(k_hnd_strided), AttentionTensorContainer(v_hnd_strided)
        ops_bench["sdpa_cudnn_bf16"] = bench_op(
            "PyTorch SDPA (BF16 Dense, Strided HND)",
            lambda: comfy_attn_mod.attention_pytorch(q_c, k_c, v_c, heads=H, skip_reshape=True),
            ref_flat=out_ref_strided,
        )
        ops_bench["sage_triton_int8qk_bf16pv"] = bench_op(
            "SageAttention 1.0.6 (INT8-QK, Strided HND)",
            lambda: comfy_attn_mod.attention_sage(q_c, k_c, v_c, heads=H, skip_reshape=True),
            ref_flat=out_ref_strided,
        )
        ops_bench["ck_dense_pure_int8"] = bench_op(
            "CK Pure-INT8 Dense (Container Prequant)",
            lambda: comfy_attn_mod.attention_comfy_kitchen_int8(
                AttentionTensorContainer(q_hnd_strided), AttentionTensorContainer(k_hnd_strided), AttentionTensorContainer(v_hnd_strided),
                heads=H, skip_reshape=True,
            ),
            ref_flat=out_ref_strided,
        )
        ops_bench["ck_sol_direct_t13_tok0"] = bench_op(
            "CK Sol-Attn Direct (tau=1.3, tok=0)",
            lambda: ck_mod.sol_attn(q_nhd_strided, k_nhd_strided, v_nhd_strided, tau=1.3, sink_blocks=[0, 75], sink_q=[67, 75], token_aug=0),
            ref_flat=out_ref_strided,
        )
        ops_bench["ck_sol_direct_t13_tok256"] = bench_op(
            "CK Sol-Attn Direct (tau=1.3, tok=256)",
            lambda: ck_mod.sol_attn(q_nhd_strided, k_nhd_strided, v_nhd_strided, tau=1.3, sink_blocks=[0, 75], sink_q=[67, 75], token_aug=256),
            ref_flat=out_ref_strided,
        )
        ops_bench["ck_sol_direct_t10_tok256"] = bench_op(
            "CK Sol-Attn Direct (tau=1.0, tok=256)",
            lambda: ck_mod.sol_attn(q_nhd_strided, k_nhd_strided, v_nhd_strided, tau=1.0, sink_blocks=[0, 75], sink_q=[67, 75], token_aug=256),
            ref_flat=out_ref_strided,
        )
        ops_bench["triton_sol_t13_int8qk_bf16pv"] = bench_op(
            "Community Triton Sol (tau=1.3, INT8-QK)",
            lambda: triton_sol_fn(q_nhd_strided, k_nhd_strided, v_nhd_strided, tau=1.3, thresh_type="diag", int8_qk=True, int8_pv=False, sink_blocks=(0, 75), sink_q=(0, 75)),
            ref_flat=out_ref_strided,
        )
        ops_bench["triton_sol_t13_int8qk_matched_sinkq"] = bench_op(
            "Community Triton Sol (tau=1.3, sink_q=67..75)",
            lambda: triton_sol_fn(q_nhd_strided, k_nhd_strided, v_nhd_strided, tau=1.3, thresh_type="diag", int8_qk=True, int8_pv=False, sink_blocks=(0, 75), sink_q=(67, 75)),
            ref_flat=out_ref_strided,
        )
        ops_bench["triton_sol_t13_int8qkpv"] = bench_op(
            "Community Triton Sol (tau=1.3, INT8-QK+PV)",
            lambda: triton_sol_fn(q_nhd_strided, k_nhd_strided, v_nhd_strided, tau=1.3, thresh_type="diag", int8_qk=True, int8_pv=True, sink_blocks=(0, 75), sink_q=(0, 75)),
            ref_flat=out_ref_strided,
        )

        sparsity_telemetry = []
        for tau_val, tok_val in [(0.8, 0), (1.0, 256), (1.3, 0), (1.3, 256), (1.5, 0)]:
            sp = measure_ck_sol_sparsity(q_nhd_strided, k_nhd_strided, v_nhd_strided, tau=tau_val, token_aug=tok_val)
            sparsity_telemetry.append(sp)
            print(
                f"  • Synthetic Sparsity Telemetry [tau={tau_val:.1f}, tok={tok_val:3d}]: "
                f"Routed={sp['mean_routed_blocks_video_q']:.1f}/{sp['total_blocks_ntb']} blocks "
                f"(Kept={sp['kept_exact_pct_video_q']:.1f}%, Sparsity={sp['sparsity_pct_video_q']:.1f}%, "
                f"ExtraTok/Grp={sp['mean_extra_tokens_routed_per_group']:.1f}, Workspace={sp['workspace_mib']:.1f} MiB)"
            )

        q_ref_warm = torch.randn((1, 16, 200, 128), generator=gen, device="cuda", dtype=torch.bfloat16)
        k_ref_warm = torch.randn((1, 16, 200, 128), generator=gen, device="cuda", dtype=torch.bfloat16)
        v_ref_warm = torch.randn((1, 16, 200, 128), generator=gen, device="cuda", dtype=torch.bfloat16)
        for ref_fn in (
            comfy_attn_mod.attention_pytorch,
            comfy_attn_mod.attention_sage,
            comfy_attn_mod.attention_comfy_kitchen_int8,
        ):
            for _ in range(2):
                _ = ref_fn(q_ref_warm, k_ref_warm, v_ref_warm, heads=16, skip_reshape=True)
        torch.cuda.synchronize()
        del q_ref_warm, k_ref_warm, v_ref_warm

    del q_hnd, k_hnd, v_hnd, qkv_fused, qf, kf, vf, q_nhd_strided, k_nhd_strided, v_nhd_strided
    del q_hnd_strided, k_hnd_strided, v_hnd_strided, out_ref_sdpa, out_ref_flat, out_ref_strided
    gc.collect()
    torch.cuda.empty_cache()

    print("\n  🔬 Verifying Discovery 1 (Chunked 2x-Pass Bootstrap vs Direct Sol-Attn) & FusedModulation Bit-Exactness...")
    from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift, rope_rotation_table
    from comfy_extras.nodes_sparse_attention import SparseAttnPatch, h3_sparse_attention
    from comfyui_sol_attn.sol_kernel.h3_fusion import fused_gate_add_, fused_modulate_, make_segment_index

    dit_model = lora_model.get_model_object("diffusion_model")
    blk0 = dit_model.blocks[0]
    x_blk = torch.randn((N, 5376), generator=gen, device="cuda", dtype=torch.bfloat16)

    t_vals_sample = torch.tensor([0.25, 0.50, 0.999], dtype=torch.float32, device="cuda")
    if getattr(dit_model, "use_adaln_curves", False):
        table = dit_model.adaln_t_table.to(device="cuda")
        pos = t_vals_sample.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb_blk = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    else:
        t_emb_blk = dit_model.time_embedder(t_vals_sample).to(torch.bfloat16)

    mod_segments_blk = [(0, 4562, 1), (4562, 4762, 2), (4762, N, 0)]
    pos_ids = torch.zeros((N, 3), device="cuda", dtype=torch.long)
    rope_freqs_blk = rope_rotation_table(dit_model.rope_freqs(pos_ids, torch.device("cuda")), torch.bfloat16)

    with torch.inference_mode():
        shift_msa, scale_msa, gate_msa, _, _, _ = blk0.adaln_proj(t_emb_blk)
        row_idx = make_segment_index(N, mod_segments_blk, x_blk.device)

        h_norm_clean = blk0.norm1(x_blk.clone()).contiguous()
        h_eager = _mod_scale_shift(h_norm_clean.clone(), shift_msa, scale_msa, mod_segments_blk)
        h_fused = fused_modulate_(h_norm_clean.clone(), shift_msa, scale_msa, row_idx)
        mod_max_abs_diff = float(torch.max(torch.abs(h_eager.float() - h_fused.float())).item())

        branch_dummy = torch.randn((N, 5376), generator=gen, device="cuda", dtype=torch.bfloat16)
        g_eager = _mod_gate(x_blk.clone(), gate_msa, branch_dummy, mod_segments_blk)
        g_fused = fused_gate_add_(x_blk.clone(), gate_msa, branch_dummy, row_idx)
        gate_max_abs_diff = float(torch.max(torch.abs(g_eager.float() - g_fused.float())).item())

        assert mod_max_abs_diff == 0.0 and gate_max_abs_diff == 0.0, (
            f"Triton H3FusedModulation not bit-exact: mod_diff={mod_max_abs_diff}, gate_diff={gate_max_abs_diff}"
        )

        h_attn_input = h_fused.clone().contiguous()
        h_scratch, g_scratch = h_norm_clean.clone(), x_blk.clone()
        for _ in range(4):
            h_scratch.copy_(h_norm_clean)
            _ = _mod_scale_shift(h_scratch, shift_msa, scale_msa, mod_segments_blk)
            _ = _mod_gate(g_scratch, gate_msa, branch_dummy, mod_segments_blk)
            h_scratch.copy_(h_norm_clean)
            _ = fused_modulate_(h_scratch, shift_msa, scale_msa, row_idx)
            _ = fused_gate_add_(g_scratch, gate_msa, branch_dummy, row_idx)
        torch.cuda.synchronize()

        t_m0 = time.perf_counter()
        for _ in range(20):
            _ = _mod_scale_shift(h_scratch, shift_msa, scale_msa, mod_segments_blk)
            _ = _mod_gate(g_scratch, gate_msa, branch_dummy, mod_segments_blk)
            _ = _mod_scale_shift(h_scratch, shift_msa, scale_msa, mod_segments_blk)
            _ = _mod_gate(g_scratch, gate_msa, branch_dummy, mod_segments_blk)
        torch.cuda.synchronize()
        eager_mod_ms = ((time.perf_counter() - t_m0) / 20.0) * 1000.0

        t_m1 = time.perf_counter()
        for _ in range(20):
            _ = fused_modulate_(h_scratch, shift_msa, scale_msa, row_idx)
            _ = fused_gate_add_(g_scratch, gate_msa, branch_dummy, row_idx)
            _ = fused_modulate_(h_scratch, shift_msa, scale_msa, row_idx)
            _ = fused_gate_add_(g_scratch, gate_msa, branch_dummy, row_idx)
        torch.cuda.synchronize()
        fused_mod_ms = ((time.perf_counter() - t_m1) / 20.0) * 1000.0

        print(
            f"  • Triton H3FusedModulation vs Eager: mod_max_diff={mod_max_abs_diff:.1e}, "
            f"gate_max_diff={gate_max_abs_diff:.1e} (Bit-Exact=True) | "
            f"Eager 4-op/blk={eager_mod_ms:.3f}ms ({eager_mod_ms * 0.4:.2f}s/400blk) vs "
            f"Triton 4-op/blk={fused_mod_ms:.3f}ms ({fused_mod_ms * 0.4:.2f}s/400blk)"
        )

        qkv_call_count = {"n": 0}
        orig_qkv_forward = blk0.attn.qkv_proj.forward

        def counted_qkv_forward(*a, **kw):
            qkv_call_count["n"] += 1
            return orig_qkv_forward(*a, **kw)

        blk0.attn.qkv_proj.forward = counted_qkv_forward
        try:
            dummy_patch = SparseAttnPatch(
                tau=1.3, topk_ratio=0.0, vsa=False, sigma_start=1.0, sigma_end=0.0,
                min_tokens=12288, dense_blocks=set(), sink_conditioning="off",
                extra_tokens=256, verbose=False,
            )
            _ = h3_sparse_attention(blk0.attn, h_attn_input, rope_freqs_blk, {}, dummy_patch, 0)
            dummy_patch.reset()
            torch.cuda.synchronize()

            qkv_call_count["n"] = 0
            t_c0 = time.perf_counter()
            out_chunked_cold = h3_sparse_attention(blk0.attn, h_attn_input, rope_freqs_blk, {}, dummy_patch, 0)
            torch.cuda.synchronize()
            chunked_cold_ms = (time.perf_counter() - t_c0) * 1000.0
            chunked_cold_qkv_calls = qkv_call_count["n"]

            qkv_call_count["n"] = 0
            t_w0 = time.perf_counter()
            for _ in range(5):
                out_chunked_warm = h3_sparse_attention(blk0.attn, h_attn_input, rope_freqs_blk, {}, dummy_patch, 0)
            torch.cuda.synchronize()
            chunked_warm_ms = ((time.perf_counter() - t_w0) / 5.0) * 1000.0
            chunked_warm_qkv_calls = qkv_call_count["n"] // 5

            def direct_attn_call():
                s = h_attn_input.shape[0]
                q, k, v = blk0.attn.qkv_proj(h_attn_input).split(H * D, dim=-1)
                q, k, v = q.view(1, s, H, D), k.view(1, s, H, D), v.view(1, s, H, D)
                qw = blk0.attn.q_norm.weight.to(device=h_attn_input.device, dtype=torch.bfloat16)
                kw = blk0.attn.k_norm.weight.to(device=h_attn_input.device, dtype=torch.bfloat16)
                ck_mod.rms_rope_split_half_(
                    q, k, rope_freqs_blk, qw, kw, epsilon=blk0.attn.q_norm.eps, rot_dim=rope_freqs_blk.shape[-3] * 2
                )
                out = ck_mod.sol_attn(q, k, v, tau=1.3, token_aug=256)
                return blk0.attn.out_proj(out.view(s, H * D))

            out_direct = direct_attn_call()
            torch.cuda.synchronize()
            qkv_call_count["n"] = 0
            t_d0 = time.perf_counter()
            for _ in range(5):
                out_direct = direct_attn_call()
            torch.cuda.synchronize()
            direct_ms = ((time.perf_counter() - t_d0) / 5.0) * 1000.0
            direct_qkv_calls = qkv_call_count["n"] // 5

            cos_cold_vs_direct, rell1_cold_vs_direct = fidelity_vs_ref(
                out_chunked_cold, out_direct.reshape(1, -1).float()
            )
            cos_warm_vs_direct, rell1_warm_vs_direct = fidelity_vs_ref(
                out_chunked_warm, out_direct.reshape(1, -1).float()
            )
        finally:
            blk0.attn.qkv_proj.forward = orig_qkv_forward

    assert chunked_cold_qkv_calls == 22
    assert chunked_warm_qkv_calls == 11
    assert direct_qkv_calls == 1

    del (
        x_blk, t_emb_blk, rope_freqs_blk, h_norm_clean, h_eager, h_fused,
        h_attn_input, h_scratch, g_scratch, branch_dummy, g_eager, g_fused,
        row_idx, out_chunked_cold, out_chunked_warm, out_direct,
    )
    gc.collect()
    torch.cuda.empty_cache()

    from comfy_extras.nodes_custom_sampler import BasicScheduler
    from comfyui_sol_attn.minimax import _TauSchedule

    sigmas_tensor = BasicScheduler.execute(lora_model, "simple", 8, 1.0)[0]
    step_sigmas = [float(s) for s in sigmas_tensor[:-1].tolist()]
    model_sampling = lora_model.get_model_object("model_sampling")
    sigma_hi = float(model_sampling.percent_to_sigma(0.0))
    sigma_lo = float(model_sampling.percent_to_sigma(1.0))

    stock_sched = _TauSchedule(1.5, 0.8, "cosine", sigma_hi, sigma_lo)
    shift_video = float(getattr(dit_model, "sigma_shift_video", 12.0))
    dense_pct = 0.20

    schedule_table = []
    for step_idx, sig in enumerate(step_sigmas):
        p_stock = stock_sched.progress(sig)
        dense_stock = p_stock < dense_pct
        tau_stock = stock_sched.tau(sig)

        t_unshifted = sig / (shift_video - (shift_video - 1.0) * sig)
        p_fixed = min(max(1.0 - t_unshifted, 0.0), 1.0)
        dense_fixed = p_fixed < dense_pct
        w_fixed = stock_sched.weight(1.0 - p_fixed)
        tau_fixed = 0.8 + (1.5 - 0.8) * w_fixed

        schedule_table.append({
            "step": step_idx,
            "sigma": round(sig, 6),
            "stock_progress": round(p_stock, 4),
            "stock_is_dense": bool(dense_stock),
            "stock_tau": round(tau_stock, 4),
            "fixed_progress": round(p_fixed, 4),
            "fixed_is_dense": bool(dense_fixed),
            "fixed_tau": round(tau_fixed, 4),
        })

    stock_dense_steps = sum(1 for r in schedule_table if r["stock_is_dense"])
    fixed_dense_steps = sum(1 for r in schedule_table if r["fixed_is_dense"])
    assert stock_dense_steps == 7 and fixed_dense_steps == 2

    return {
        "shape_B_H_N_D": [B, H, N, D],
        "flops_per_call_tflops": round(flops / 1e12, 3),
        "operators": ops_bench,
        "sparsity_telemetry": sparsity_telemetry,
        "discovery1_block_benchmark": {
            "chunked_cold_first_step_ms": round(chunked_cold_ms, 3),
            "chunked_cold_qkv_calls_per_block": chunked_cold_qkv_calls,
            "chunked_warm_step_ms": round(chunked_warm_ms, 3),
            "chunked_warm_qkv_calls_per_block": chunked_warm_qkv_calls,
            "direct_zero_copy_step_ms": round(direct_ms, 3),
            "direct_qkv_calls_per_block": direct_qkv_calls,
            "cosine_sim_chunked_cold_vs_direct": cos_cold_vs_direct,
            "rel_l1_chunked_cold_vs_direct": rell1_cold_vs_direct,
            "cosine_sim_chunked_warm_vs_direct": cos_warm_vs_direct,
            "rel_l1_chunked_warm_vs_direct": rell1_warm_vs_direct,
            "est_6_sparse_steps_chunked_sec": round((chunked_cold_ms * 50 + chunked_warm_ms * 250) / 1000.0, 2),
            "est_6_sparse_steps_direct_sec": round((direct_ms * 300) / 1000.0, 2),
        },
        "fused_modulation_verification": {
            "mod_max_abs_diff_vs_eager": mod_max_abs_diff,
            "gate_max_abs_diff_vs_eager": gate_max_abs_diff,
            "is_bit_exact": bool(mod_max_abs_diff == 0.0 and gate_max_abs_diff == 0.0),
            "eager_4ops_per_block_ms": round(eager_mod_ms, 3),
            "fused_4ops_per_block_ms": round(fused_mod_ms, 3),
            "est_400_blocks_saved_sec": round((eager_mod_ms - fused_mod_ms) * 0.4, 2),
        },
        "discovery2_schedule_verification": {
            "sigma_shift_video": shift_video,
            "dense_percent": dense_pct,
            "stock_dense_steps_of_8": stock_dense_steps,
            "fixed_dense_steps_of_8": fixed_dense_steps,
            "steps": schedule_table,
        },
    }
