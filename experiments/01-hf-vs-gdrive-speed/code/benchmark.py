#!/usr/bin/env python3
"""
Experiment 01: H3-Base-FL2VA End-to-End Pipeline Benchmark (42.03 GB, 5 files)
Tests the complete real-world loading chain on Google Colab:
  Stage 1: Cloud Storage (Google Drive or Hugging Face) -> Local Disk (/content)
  Stage 2: Disk -> System RAM (safetensors.torch.load_file(..., device="cpu"))
  Stage 3: System RAM -> GPU VRAM (tensor.to("cuda") + torch.cuda.synchronize())

Compares 3 end-to-end pipelines:
  - Pipeline A : Hugging Face (hf_transfer) -> Download to /content -> Load to RAM -> Move to GPU VRAM
  - Pipeline B1: Google Drive -> Copy to /content -> Load to RAM -> Move to GPU VRAM
  - Pipeline B2: Google Drive (Zero-Copy in-place) -> Load directly from /content/drive to RAM -> Move to GPU VRAM
"""

import os
import sys

# Must be set BEFORE importing huggingface_hub:
# 1. Enable Rust Xet high-performance transfer (and legacy hf_transfer fallback)
# 2. Disable implicit token lookup (prevents Colab UI secret vault polling in headless colab-cli sessions)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import argparse
import gc
import importlib.util
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure required packages (especially hf_transfer and safetensors) are installed on fresh Colab VMs
for _pkg in ("hf_transfer", "huggingface_hub", "safetensors", "psutil"):
    if importlib.util.find_spec(_pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)

REPO_ID = "Comfy-Org/MiniMax-H3"

# 组合 1：官方默认黄金组合（共 5 个文件，总计 42.03 GB / 39.14 GiB）
GOLD_BUNDLE_FILES: List[Dict[str, Any]] = [
    {
        "role": "FL2VA 生成主干 (Pruned INT8 ConvRot)",
        "path": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "size": 20970379616,  # 20.97 GB
    },
    {
        "role": "文本编码器 (Qwen3-VL-32B NVFP4 AWQ)",
        "path": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "size": 15687142551,  # 15.69 GB
    },
    {
        "role": "视频解码器 (Video VAE INT8 ConvRot)",
        "path": "vae/minimax_h3_video_vae_int8_convrot.safetensors",
        "size": 2811065184,   # 2.81 GB
    },
    {
        "role": "8步加速 LoRA (FL2V Turbo 8-Step)",
        "path": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "size": 1956193000,   # 1.96 GB
    },
    {
        "role": "音频解码器 (Audio VAE FP32)",
        "path": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "size": 605254808,    # 0.61 GB
    },
]


def free_space_gib(path: Path = Path("/content")) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def drop_linux_page_cache() -> None:
    """
    Drop Linux kernel page/dentry/inode caches (/proc/sys/vm/drop_caches = 3).
    NOTE: Never rmtree /root/.config/Google/DriveFS while /content/drive is mounted,
    as deleting DriveFS's internal cache directory crashes the live drive-fs FUSE daemon!
    """
    subprocess.run(["sync"], check=False)
    if os.geteuid() == 0 and Path("/proc/sys/vm/drop_caches").exists():
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
        except Exception:
            pass


def print_hardware_inspection() -> None:
    import psutil
    import torch

    disk_total = shutil.disk_usage("/content").total / (1024 ** 3)
    disk_free = free_space_gib()
    ram_total = psutil.virtual_memory().total / (1024 ** 3)
    ram_avail = psutil.virtual_memory().available / (1024 ** 3)

    print("=" * 84)
    print("🔍 当前 Colab 虚拟机【① 磁盘 -> ② 内存 -> ③ 显存】硬件体检：")
    print(f"  ① 本地磁盘 (Disk /content) : 总计 {disk_total:.1f} GiB | 可用 {disk_free:.1f} GiB")
    print(f"  ② 系统内存 (System RAM)    : 总计 {ram_total:.1f} GiB | 可用 {ram_avail:.1f} GiB")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gib = props.total_memory / (1024 ** 3)
        print(f"  ③ GPU 显存 (VRAM)          : {props.name} | 总计 {vram_gib:.1f} GiB")
        if vram_gib >= 45.0:
            print("     👑 检测到超大显存 GPU (如 G4 RTX PRO 6000 96GB / A100-80G / H100)，支持 42.03GB 全套常驻显存！")
        else:
            print("     💡 当前显存 < 45 GiB，脚本将按实际推理顺序逐模块测试 [Disk -> RAM -> VRAM] 并释放显存。")
    else:
        print("  ③ GPU 显存 (VRAM)          : 未挂载 GPU (当前为 CPU 模式，阶段 ③ RAM->VRAM 将跳过)")
    print("=" * 84)


def measure_disk_to_ram_and_vram(file_path: Path) -> tuple[float, float]:
    """
    Given a .safetensors file on disk (/content or /content/drive):
      1. Measure loading from Disk -> System RAM (CPU tensors)
      2. If CUDA is available, measure moving CPU tensors -> GPU VRAM (CUDA tensors)
      3. Free memory cleanly after timing so sequential modules never OOM.
    """
    import torch
    from safetensors.torch import load_file

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Stage 2: Disk -> System RAM
    t0 = time.perf_counter()
    cpu_state_dict = load_file(str(file_path), device="cpu")
    ram_load_sec = time.perf_counter() - t0

    # Stage 3: System RAM -> GPU VRAM
    vram_move_sec = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gpu_state_dict = {k: v.to("cuda", non_blocking=False) for k, v in cpu_state_dict.items()}
        torch.cuda.synchronize()
        vram_move_sec = time.perf_counter() - t1
        del gpu_state_dict
        torch.cuda.empty_cache()

    del cpu_state_dict
    gc.collect()
    return ram_load_sec, vram_move_sec


def download_single_file_hf(rel_path: str, dest_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    out_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=rel_path,
            local_dir=str(dest_root),
            force_download=True,
        )
    )
    shutil.rmtree(dest_root / ".cache", ignore_errors=True)
    return out_path


def benchmark_pipeline_a_and_sync_gdrive(local_dest: Path, gdrive_model_root: Path) -> Dict[str, Any]:
    """
    1. Run Pipeline A: Hugging Face (hf_transfer) -> Download to /content -> Load to RAM -> Move to VRAM
    2. For any file not yet persisted in Google Drive, copy the local file to Google Drive (outside timer)
       so we only download from Hugging Face ONCE!
    """
    print("\n" + "=" * 84)
    print("▶️ [链路 A] Hugging Face (hf_transfer) -> 下载至 /content 本地盘 -> 加载至 RAM -> 搬运至 VRAM")
    print("=" * 84)

    shutil.rmtree(local_dest, ignore_errors=True)
    local_dest.mkdir(parents=True, exist_ok=True)
    gdrive_model_root.mkdir(parents=True, exist_ok=True)
    drop_linux_page_cache()

    total_bytes = 0
    tx_sec_total = 0.0
    ram_sec_total = 0.0
    vram_sec_total = 0.0

    for idx, item in enumerate(GOLD_BUNDLE_FILES, 1):
        rel_path = item["path"]
        fsize = item["size"]

        # ① Timed Download: HF -> /content
        t0 = time.perf_counter()
        dst_file = download_single_file_hf(rel_path, local_dest)
        tx_sec = time.perf_counter() - t0

        drop_linux_page_cache()

        # ② Timed Disk -> RAM & ③ Timed RAM -> VRAM
        ram_sec, vram_sec = measure_disk_to_ram_and_vram(dst_file)

        tx_sec_total += tx_sec
        ram_sec_total += ram_sec
        vram_sec_total += vram_sec
        total_bytes += fsize

        tx_mbs = (fsize / (1024 ** 2)) / tx_sec
        print(
            f"  [A  {idx}/5] {item['role']} ({fsize/1e9:.2f} GB) | "
            f"①HF下载: {tx_sec:.1f}s ({tx_mbs:.0f} MiB/s) | ②入RAM: {ram_sec:.1f}s | ③入VRAM: {vram_sec:.1f}s"
        )

        # Untimed: Persist to Google Drive if not already present in Google Drive
        gdrive_file = gdrive_model_root / rel_path
        if not (gdrive_file.exists() and gdrive_file.stat().st_size == fsize):
            print(f"      ☁️ 正在将该文件同步备份至 Google Drive ({gdrive_file})...")
            gdrive_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst_file, gdrive_file)
            subprocess.run(["sync"], check=False)
            print("      ✅ 已持久化至 Google Drive")
        else:
            print("      ⏭️ Google Drive 中已存在该文件，跳过备份")

        # Delete local copy after syncing to keep /content disk clean
        dst_file.unlink(missing_ok=True)

    shutil.rmtree(local_dest, ignore_errors=True)
    drop_linux_page_cache()

    total_sec = tx_sec_total + ram_sec_total + vram_sec_total
    tx_avg_mbs = (total_bytes / (1024 ** 2)) / tx_sec_total
    print(f"\n✅ [链路 A 完成] HF直连总耗时: {total_sec:.1f}s ({total_sec/60:.2f} min) | 其中下载 {tx_sec_total:.1f}s ({tx_avg_mbs:.0f} MiB/s)")
    return {
        "name": "链路 A : Hugging Face (Xet) -> /content -> RAM -> VRAM",
        "tx_sec": tx_sec_total,
        "tx_mbs": tx_avg_mbs,
        "ram_sec": ram_sec_total,
        "vram_sec": vram_sec_total,
        "total_sec": total_sec,
    }


def benchmark_pipeline_b1_gdrive_copy(gdrive_model_root: Path, local_dest: Path) -> Dict[str, Any]:
    """
    Pipeline B1: Google Drive -> Copy to /content -> Load to RAM -> Move to VRAM
    """
    print("\n" + "=" * 84)
    print("▶️ [链路 B1] Google Drive -> 拷贝至 /content 本地盘 -> 加载至 RAM -> 搬运至 VRAM")
    print("=" * 84)

    shutil.rmtree(local_dest, ignore_errors=True)
    local_dest.mkdir(parents=True, exist_ok=True)
    drop_linux_page_cache()

    total_bytes = 0
    tx_sec_total = 0.0
    ram_sec_total = 0.0
    vram_sec_total = 0.0

    for idx, item in enumerate(GOLD_BUNDLE_FILES, 1):
        rel_path = item["path"]
        fsize = item["size"]
        src_file = gdrive_model_root / rel_path
        dst_file = local_dest / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if not src_file.exists() or src_file.stat().st_size != fsize:
            raise FileNotFoundError(f"Google Drive 缺失文件: {src_file}")

        # ① Copy Drive -> /content
        t0 = time.perf_counter()
        shutil.copy2(src_file, dst_file)
        tx_sec = time.perf_counter() - t0

        drop_linux_page_cache()

        # ② Disk -> RAM & ③ RAM -> VRAM
        ram_sec, vram_sec = measure_disk_to_ram_and_vram(dst_file)
        dst_file.unlink(missing_ok=True)

        tx_sec_total += tx_sec
        ram_sec_total += ram_sec
        vram_sec_total += vram_sec
        total_bytes += fsize

        tx_mbs = (fsize / (1024 ** 2)) / tx_sec
        print(
            f"  [B1 {idx}/5] {item['role']} ({fsize/1e9:.2f} GB) | "
            f"①拷贝: {tx_sec:.1f}s ({tx_mbs:.0f} MiB/s) | ②入RAM: {ram_sec:.1f}s | ③入VRAM: {vram_sec:.1f}s"
        )

    shutil.rmtree(local_dest, ignore_errors=True)
    drop_linux_page_cache()

    total_sec = tx_sec_total + ram_sec_total + vram_sec_total
    tx_avg_mbs = (total_bytes / (1024 ** 2)) / tx_sec_total
    print(f"\n✅ [链路 B1 完成] Drive拷贝流总耗时: {total_sec:.1f}s ({total_sec/60:.2f} min) | 其中拷贝 {tx_sec_total:.1f}s ({tx_avg_mbs:.0f} MiB/s)")
    return {
        "name": "链路 B1: Google Drive -> 拷贝至 /content -> RAM -> VRAM",
        "tx_sec": tx_sec_total,
        "tx_mbs": tx_avg_mbs,
        "ram_sec": ram_sec_total,
        "vram_sec": vram_sec_total,
        "total_sec": total_sec,
    }


def benchmark_pipeline_b2_gdrive_inplace(gdrive_model_root: Path) -> Dict[str, Any]:
    """
    Pipeline B2: Google Drive (Zero-Copy in-place) -> Load directly from /content/drive to RAM -> Move to VRAM
    """
    print("\n" + "=" * 84)
    print("▶️ [链路 B2] Google Drive 原地零拷贝 -> 直接从 /content/drive 加载至 RAM -> 搬运至 VRAM")
    print("=" * 84)

    drop_linux_page_cache()

    total_bytes = 0
    ram_sec_total = 0.0
    vram_sec_total = 0.0

    for idx, item in enumerate(GOLD_BUNDLE_FILES, 1):
        rel_path = item["path"]
        fsize = item["size"]
        src_file = gdrive_model_root / rel_path

        drop_linux_page_cache()
        ram_sec, vram_sec = measure_disk_to_ram_and_vram(src_file)

        ram_sec_total += ram_sec
        vram_sec_total += vram_sec
        total_bytes += fsize

        equiv_mbs = (fsize / (1024 ** 2)) / ram_sec
        print(
            f"  [B2 {idx}/5] {item['role']} ({fsize/1e9:.2f} GB) | "
            f"①零拷贝 | ②直读入RAM: {ram_sec:.1f}s ({equiv_mbs:.0f} MiB/s) | ③入VRAM: {vram_sec:.1f}s"
        )
        drop_linux_page_cache()

    total_sec = ram_sec_total + vram_sec_total
    equiv_avg_mbs = (total_bytes / (1024 ** 2)) / ram_sec_total
    print(f"\n✅ [链路 B2 完成] Drive零拷贝直读总耗时: {total_sec:.1f}s ({total_sec/60:.2f} min)")
    return {
        "name": "链路 B2: Google Drive 原地零拷贝 -> RAM -> VRAM",
        "tx_sec": 0.0,
        "tx_mbs": equiv_avg_mbs,
        "ram_sec": ram_sec_total,
        "vram_sec": vram_sec_total,
        "total_sec": total_sec,
    }


def print_summary(results: List[Dict[str, Any]]) -> None:
    best = min(results, key=lambda r: r["total_sec"])
    print("\n" + "=" * 105)
    print("📊 实验 01 全链路实测报告：FL2VA 黄金组合 (42.03 GB) 【云端 -> 本地盘 -> 内存 RAM -> 显存 VRAM】")
    print("=" * 105)
    print("| 端到端链路方案 | ① 落盘耗时 (吞吐率) | ② Disk -> RAM 耗时 | ③ RAM -> VRAM 耗时 | **三阶段累计耗时** |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for r in results:
        crown = " 🏆" if r is best else ""
        tx_str = "0.0 s (零拷贝)" if r["tx_sec"] == 0.0 else f"{r['tx_sec']:.1f} s ({r['tx_mbs']:.0f} MiB/s)"
        print(
            f"| **{r['name']}{crown}** | {tx_str} | "
            f"{r['ram_sec']:.1f} s | {r['vram_sec']:.1f} s | "
            f"**{r['total_sec']:.1f} s ({r['total_sec']/60:.2f} min)** |"
        )
    print("=" * 105)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="42.03GB FL2VA End-to-End (Disk -> RAM -> VRAM) Benchmark")
    parser.add_argument(
        "--gdrive-model-root",
        type=Path,
        default=Path("/content/drive/MyDrive/models/Comfy-MiniMax-H3-FL2VA"),
        help="Google Drive 中保存 5 个核心权重的根目录",
    )
    parser.add_argument(
        "--local-dest",
        type=Path,
        default=Path("/content/Comfy-MiniMax-H3-FL2VA"),
        help="Colab 本地磁盘 (/content) 目标目录",
    )
    args, _ = parser.parse_known_args()

    print_hardware_inspection()
    res_a = benchmark_pipeline_a_and_sync_gdrive(args.local_dest, args.gdrive_model_root)
    res_b1 = benchmark_pipeline_b1_gdrive_copy(args.gdrive_model_root, args.local_dest)
    res_b2 = benchmark_pipeline_b2_gdrive_inplace(args.gdrive_model_root)
    print_summary([res_b1, res_b2, res_a])


if __name__ == "__main__":
    main()
