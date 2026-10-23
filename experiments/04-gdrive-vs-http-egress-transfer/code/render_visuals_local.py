#!/usr/bin/env python3
"""
Local Visualization Renderer for Experiment 04 (0 CU — Executes on Local Mac):
Generates two high-resolution diagnostic figures:
  1. outputs/egress_gpu_time_and_cu_comparison.png
  2. outputs/bandwidth_sensitivity_and_protocol_bloat.png
"""

import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "code" else _SCRIPT_DIR
OUTPUT_DIR = ROOT_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "exp04_egress_transfer_metrics.json"
LEDGER_EXCERPT_PATH = OUTPUT_DIR / "colab_session_ledger_excerpt.json"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    ledger_excerpt = json.loads(LEDGER_EXCERPT_PATH.read_text())

    # Synchronize settled ledger audit into metrics JSON
    metrics["session_ledger"]["assign_to_unassign_http_wall_sec"] = 92.918
    metrics["session_ledger"]["theoretical_session_cu_at_7_2046_over_http_span"] = 0.1859
    metrics["session_ledger"]["post_session_settled_verification_ccu_info"] = ledger_excerpt["events"][-1]
    metrics["session_ledger"]["settled_ledger_deducted_cu"] = 0.0
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    comp = metrics["comparison_by_tier"]
    tiers = ["T1_Single_Run", "T2_4Run_Batch", "T3_12Run_Exp03", "T4_36Run_Sweep"]
    tier_labels = [
        "T1: 单视频产出\n(10.63 MB raw\n14.55 MB wire)",
        "T2: 4视频小批量\n(41.09 MB raw\n56.23 MB wire)",
        "T3: 实验03全量12视频\n(122.42 MB raw\n167.52 MB wire)",
        "T4: 36视频大网格扫描\n(367.26 MB raw\n502.56 MB wire)",
    ]

    # -------------------------------------------------------------------------
    # Figure 1: GPU Billable Time / CU Cost vs. Total Wall-Clock Decomposition
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=220)
    fig.patch.set_facecolor("#F8FAFC")
    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E1")
        ax.set_axisbelow(True)

    x = np.arange(len(tiers))
    w = 0.27

    http_gpu_sec = [comp[t]["method_A_http_direct"]["gpu_billable_sec"] for t in tiers]
    fuse_gpu_sec = [comp[t]["method_B1_gdrive_fuse_mount_ref"]["total_gpu_sec_incl_daemon_init"] for t in tiers]
    relay_gpu_sec = [comp[t]["method_B2_gdrive_rest_relay"]["stage1_gpu_billable_sec"] for t in tiers]

    # Also plot Exp 03 actual T3 measurement marker (26.598s)
    b_http = ax1.bar(
        x - w,
        http_gpu_sec,
        width=w,
        color="#EF4444",
        edgecolor="#991B1B",
        linewidth=1.0,
        label="方案 A: HTTP Jupyter 直拉 (colab download, +36.84% Base64 膨胀)",
    )
    b_fuse = ax1.bar(
        x,
        fuse_gpu_sec,
        width=w,
        color="#F59E0B",
        edgecolor="#92400E",
        linewidth=1.0,
        label="方案 B1: 传统 FUSE 挂载 (/content/drive 守护进程初始化 + flush_and_unmount)",
    )
    b_relay = ax1.bar(
        x + w,
        relay_gpu_sec,
        width=w,
        color="#10B981",
        edgecolor="#065F46",
        linewidth=1.2,
        label="方案 B2 (推荐): Google Drive v3 API 云端内网直传 -> 立即 colab stop",
    )

    # Annotate Exp 03 historical 26.60s point on T3
    ax1.scatter([2 - w], [26.598], color="#7F1D1D", s=70, zorder=6, marker="D", label="实验 03 实测 HTTP 直拉耗时 (26.60s / 0.0532 CU)")
    ax1.plot([2 - w, 2 - w], [http_gpu_sec[2], 26.598], color="#7F1D1D", linestyle=":", linewidth=1.5, zorder=5)
    ax1.annotate(
        "实验03实测: 26.60s\n(0.0532 CU, 节省 91.5%)",
        xy=(2 - w, 26.598),
        xytext=(2 - w - 0.65, 32.0),
        fontsize=8.2,
        color="#7F1D1D",
        arrowprops=dict(arrowstyle="->", color="#7F1D1D", lw=1.0),
    )

    for grp_idx, (bars, color_txt) in enumerate([(b_http, "#991B1B"), (b_fuse, "#92400E"), (b_relay, "#065F46")]):
        for bar_idx, bar in enumerate(bars):
            h = bar.get_height()
            cu = h * 7.2046 / 3600.0
            y_off = 0.7
            if bar_idx == 2 and grp_idx == 0:
                y_off = 0.5
            elif bar_idx == 2 and grp_idx == 1:
                y_off = 3.2
            elif bar_idx == 3 and grp_idx == 1:
                y_off = 1.0
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + y_off,
                f"{h:.2f}s\n({cu:.4f}CU)",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color=color_txt,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_labels, fontsize=9.0)
    ax1.set_ylabel("占用 G4 GPU 计费时长 (秒) & 消耗算力 (CU)", fontsize=10.5)
    ax1.set_title("A. 四档位输出包：占用 G4 GPU 计费时长与 CU 消耗对比 (越低越省)", fontsize=11.5, pad=12)
    ax1.set_ylim(0, 56)
    ax1.legend(loc="upper left", fontsize=8.2, framealpha=0.95)

    # Panel B: Total End-to-End Wall-Clock Time Decomposition (Stage 1 Billable GPU vs Stage 2 0-CU Local Pull)
    stage2_par8_sec = [comp[t]["method_B2_gdrive_rest_relay"]["stage2_off_gpu_par8_sec"] for t in tiers]

    ax2.bar(
        x - 0.21,
        http_gpu_sec,
        width=0.35,
        color="#EF4444",
        edgecolor="#991B1B",
        linewidth=1.0,
        label="方案 A: HTTP 直拉 (全程占用 G4 GPU 计费)",
    )
    ax2.bar(
        x + 0.21,
        relay_gpu_sec,
        width=0.35,
        color="#10B981",
        edgecolor="#065F46",
        linewidth=1.0,
        label="方案 B2 阶段1: Colab G4 -> Google Drive (唯一计费阶段)",
    )
    ax2.bar(
        x + 0.21,
        stage2_par8_sec,
        bottom=relay_gpu_sec,
        width=0.35,
        color="#3B82F6",
        edgecolor="#1D4ED8",
        linewidth=1.0,
        hatch="//",
        alpha=0.88,
        label="方案 B2 阶段2: Google Drive -> 本地 Mac 8线程并发拉取 (已停机 = 0.00 CU!)",
    )

    for idx in range(len(tiers)):
        tot_a = http_gpu_sec[idx]
        tot_b = relay_gpu_sec[idx] + stage2_par8_sec[idx]
        sav_pct = comp[tiers[idx]]["method_B2_gdrive_rest_relay"]["gpu_cu_saving_pct_vs_http"]
        y_off_a = 0.8 if idx != 3 else 0.6
        y_off_b = 0.8 if idx not in (0, 3) else 3.2
        ax2.text(
            x[idx] - 0.21,
            tot_a + y_off_a,
            f"总{tot_a:.2f}s\n(GPU {tot_a:.2f}s)",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#991B1B",
        )
        ax2.text(
            x[idx] + 0.21,
            tot_b + y_off_b,
            f"总{tot_b:.2f}s\nGPU仅{relay_gpu_sec[idx]:.2f}s (-{sav_pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#065F46",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_labels, fontsize=9.0)
    ax2.set_ylabel("端到端总墙钟耗时 (秒)", fontsize=10.5)
    ax2.set_title("B. 端到端总耗时拆解：GPU 计费时长 vs. 停机后 0 CU 本地拉取时长", fontsize=11.5, pad=12)
    ax2.set_ylim(0, 58)
    ax2.legend(loc="upper left", fontsize=8.2, framealpha=0.95)

    plt.tight_layout()
    out1 = OUTPUT_DIR / "egress_gpu_time_and_cu_comparison.png"
    fig.savefig(out1, dpi=220)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Local Bandwidth Sensitivity & Protocol Inflation / Backbone Scaling
    # -------------------------------------------------------------------------
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16.5, 6.6), dpi=220)
    fig2.patch.set_facecolor("#F8FAFC")
    for ax in (ax3, ax4):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
        ax.set_axisbelow(True)

    bw_list = [row["local_wire_bandwidth_mb_s"] for row in metrics["bandwidth_sensitivity_t3"]]
    http_cu_list = [row["http_direct_cu"] for row in metrics["bandwidth_sensitivity_t3"]]
    gdrive_cu_list = [row["gdrive_relay_cu"] for row in metrics["bandwidth_sensitivity_t3"]]
    http_sec_list = [row["http_direct_gpu_sec"] for row in metrics["bandwidth_sensitivity_t3"]]
    gdrive_wall_1w_list = [row["gdrive_relay_wall_1stream_sec"] for row in metrics["bandwidth_sensitivity_t3"]]

    ax3.plot(
        bw_list,
        http_sec_list,
        marker="o",
        linewidth=2.4,
        color="#EF4444",
        label="方案 A (HTTP 直拉): GPU 计费耗时 = 端到端总耗时 (受本地带宽严格阻塞)",
    )
    ax3.plot(
        bw_list,
        gdrive_wall_1w_list,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#3B82F6",
        label="方案 B2 (GDrive 中转, 单流拉取): 端到端总墙钟耗时 (零 Base64 膨胀, <30.2 MB/s 时总耗时反而更快!)",
    )
    ax3.axhline(
        y=2.2659,
        color="#10B981",
        linewidth=2.8,
        label="方案 B2 (GDrive 中转): G4 GPU 唯一计费耗时恒定 2.27s (0.0045 CU, 与本地慢网完全解耦!)",
    )

    ax3.axvline(x=6.30, color="#7F1D1D", linestyle=":", linewidth=1.4)
    ax3.annotate(
        "实验03本地带宽 (6.30 MB/s wire)\nHTTP直拉: 26.60s (0.0532 CU)\nGDrive中转GPU仅: 2.27s (省 91.5%)",
        xy=(6.30, 26.60),
        xytext=(9.5, 115.0),
        fontsize=8.5,
        fontweight="bold",
        color="#7F1D1D",
        arrowprops=dict(arrowstyle="->", color="#7F1D1D", lw=1.2),
    )

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xticks([0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
    ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax3.set_xlabel("用户本地网络下载速率 (MB/s, 对数坐标)", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("耗时 (秒, 对数坐标)", fontsize=10.5, fontweight="bold")
    ax3.set_title("A. 本地网络速率敏感度模型 (以实验03的 122.42 MB 归档包为例)", fontsize=11.5, fontweight="bold", pad=12)
    ax3.legend(loc="upper right", fontsize=8.1, framealpha=0.95)

    # Panel B: Datacenter Backbone Upload Throughput vs HTTP Direct & Stage-2 Parallel Pull Throughput
    raw_mbs = [comp[t]["tier_meta"]["raw_mb"] for t in tiers]
    dc_media_tp = [ metrics["remote_probe"]["gdrive_uploads"][t]["media_mean_mb_s"] for t in tiers ]
    dc_resum_tp = [ metrics["remote_probe"]["gdrive_uploads"][t]["resumable_mean_mb_s"] for t in tiers ]
    stage2_par8_tp = [ comp[t]["method_B2_gdrive_rest_relay"]["stage2_off_gpu_par8_mb_s"] for t in tiers ]
    http_eff_tp = [ comp[t]["method_A_http_direct"]["effective_payload_mb_s"] for t in tiers ]

    ax4.plot(
        raw_mbs,
        dc_media_tp,
        marker="o",
        linewidth=2.5,
        color="#10B981",
        label="阶段1: Colab G4 -> Google Drive (REST Media 单次直传, Google 机房内网骨干网)",
    )
    ax4.plot(
        raw_mbs,
        dc_resum_tp,
        marker="^",
        linewidth=2.0,
        linestyle="--",
        color="#059669",
        label="阶段1: Colab G4 -> Google Drive (REST Resumable 断点续传协议)",
    )
    ax4.plot(
        raw_mbs,
        stage2_par8_tp,
        marker="s",
        linewidth=2.2,
        color="#3B82F6",
        label="阶段2: Google Drive -> 本地 Mac (8线程 HTTP Range 并发拉取, 0.00 CU)",
    )
    ax4.plot(
        raw_mbs,
        http_eff_tp,
        marker="D",
        linewidth=2.2,
        color="#EF4444",
        label="方案A: Colab G4 -> 本地 Mac (colab download 有效二进制吞吐率, 受 +36.84% Base64 拖累)",
    )

    for rx, y_dc, y_http in zip(raw_mbs, dc_media_tp, http_eff_tp):
        ax4.annotate(f"{y_dc:.1f} MB/s", (rx, y_dc), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.3, fontweight="bold", color="#065F46")
        ax4.annotate(f"{y_http:.2f} MB/s", (rx, y_http), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.0, fontweight="bold", color="#991B1B")

    ax4.set_xlabel("归档包原始二进制体积 (MB)", fontsize=10.5, fontweight="bold")
    ax4.set_ylabel("有效二进制吞吐率 (MB/s, 越高越快)", fontsize=10.5, fontweight="bold")
    ax4.set_title("B. 吞吐率随文件体积扩展曲线：Google 机房内网 (79.9 MB/s) vs. 本地直拉", fontsize=11.5, fontweight="bold", pad=12)
    ax4.set_ylim(0, 95)
    ax4.legend(loc="upper left", fontsize=8.1, framealpha=0.95)

    plt.tight_layout()
    out2 = OUTPUT_DIR / "bandwidth_sensitivity_and_protocol_bloat.png"
    fig2.savefig(out2, dpi=220)
    plt.close(fig2)
    print(f"Rendered:\n  - {out1}\n  - {out2}")


if __name__ == "__main__":
    main()
