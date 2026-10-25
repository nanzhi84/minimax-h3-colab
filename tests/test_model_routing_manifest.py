#!/usr/bin/env python3
"""
Black-box contract tests for the 2x2 BF16 model routing matrix (fl2va / ref2va x quality / fast),
manifest schema validation, and local 0-CU batch planner packing (`tests/test_model_routing_manifest.py`).
"""

import json
import struct
import tarfile
import tempfile
import unittest
import zlib
from pathlib import Path

from studio.api import local_planner, manifest_schema


def _make_valid_rgb_png_bytes(r: int, g: int, b: int) -> bytes:
    """Create a real, valid 1x1 RGB PNG byte sequence without external library dependencies."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    raw_scanline = bytes([0, r & 0xFF, g & 0xFF, b & 0xFF])
    idat_data = zlib.compress(raw_scanline)
    idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
    idat_chunk = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)

    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return signature + ihdr_chunk + idat_chunk + iend_chunk


class TestRouteProfileResolver(unittest.TestCase):
    def test_four_canonical_bf16_profiles_match_2x2_contract(self) -> None:
        self.assertTrue(
            hasattr(manifest_schema, "ROUTE_PROFILES_2X2"),
            "manifest_schema must expose ROUTE_PROFILES_2X2",
        )
        profiles = manifest_schema.ROUTE_PROFILES_2X2
        self.assertEqual(
            set(profiles.keys()),
            {
                "fl2va_bf16_quality",
                "fl2va_bf16_fast",
                "ref2va_bf16_quality",
                "ref2va_bf16_fast",
            },
        )

        p_fl_q = profiles["fl2va_bf16_quality"]
        self.assertEqual(p_fl_q["model_route"], "fl2va")
        self.assertEqual(p_fl_q["profile"], "quality")
        self.assertEqual(p_fl_q["precision"], "bf16")
        self.assertEqual(p_fl_q["unet_name"], "minimax_h3_fl2va_pruned_bf16.safetensors")
        self.assertEqual(p_fl_q["clip_name"], "qwen3vl_32b_minimax_h3_bf16.safetensors")
        self.assertEqual(p_fl_q["video_vae_name"], "minimax_h3_video_vae_fp16.safetensors")
        self.assertEqual(p_fl_q["audio_vae_name"], "minimax_h3_audio_vae_fp32.safetensors")
        self.assertIsNone(p_fl_q["lora_name"])
        self.assertEqual(p_fl_q["default_steps"], 20)
        self.assertEqual(p_fl_q["default_scheduler"], "simple")
        self.assertEqual(p_fl_q["dense_backend"], "sdpa")
        self.assertEqual(p_fl_q["sparse_mode"], "none")
        self.assertIs(p_fl_q["fused_mod"], True)

        p_fl_f = profiles["fl2va_bf16_fast"]
        self.assertEqual(p_fl_f["model_route"], "fl2va")
        self.assertEqual(p_fl_f["profile"], "fast")
        self.assertEqual(p_fl_f["precision"], "bf16")
        self.assertEqual(p_fl_f["unet_name"], "minimax_h3_fl2va_pruned_bf16.safetensors")
        self.assertEqual(
            p_fl_f["lora_name"],
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        )
        self.assertEqual(p_fl_f["default_steps"], 8)
        self.assertEqual(p_fl_f["default_scheduler"], "simple")
        self.assertEqual(p_fl_f["dense_backend"], "sage")
        self.assertEqual(p_fl_f["sparse_mode"], "ck_sol_direct")
        self.assertAlmostEqual(float(p_fl_f["tau"]), 1.3)
        self.assertEqual(p_fl_f["extra_tokens"], 256)
        self.assertAlmostEqual(float(p_fl_f["start_percent"]), 0.20)
        self.assertIs(p_fl_f["fused_mod"], True)

        p_ref_q = profiles["ref2va_bf16_quality"]
        self.assertEqual(p_ref_q["model_route"], "ref2va")
        self.assertEqual(p_ref_q["profile"], "quality")
        self.assertEqual(p_ref_q["precision"], "bf16")
        self.assertEqual(p_ref_q["unet_name"], "minimax_h3_ref2va_pruned_bf16.safetensors")
        self.assertIsNone(p_ref_q["lora_name"])
        self.assertEqual(p_ref_q["default_steps"], 20)
        self.assertEqual(p_ref_q["default_scheduler"], "beta")
        self.assertEqual(p_ref_q["dense_backend"], "sdpa")
        self.assertEqual(p_ref_q["sparse_mode"], "none")
        self.assertIs(p_ref_q["fused_mod"], True)

        p_ref_f = profiles["ref2va_bf16_fast"]
        self.assertEqual(p_ref_f["model_route"], "ref2va")
        self.assertEqual(p_ref_f["profile"], "fast")
        self.assertEqual(p_ref_f["precision"], "bf16")
        self.assertEqual(p_ref_f["unet_name"], "minimax_h3_ref2va_pruned_bf16.safetensors")
        self.assertEqual(
            p_ref_f["lora_name"],
            "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors",
        )
        self.assertEqual(p_ref_f["default_steps"], 8)
        self.assertEqual(p_ref_f["default_scheduler"], "simple")
        self.assertEqual(p_ref_f["dense_backend"], "sage")
        self.assertEqual(p_ref_f["sparse_mode"], "ck_sol_direct")
        self.assertAlmostEqual(float(p_ref_f["tau"]), 1.0)
        self.assertEqual(p_ref_f["extra_tokens"], 256)
        self.assertEqual(p_ref_f["dense_blocks"], "0,1,48,49")
        self.assertAlmostEqual(float(p_ref_f["start_percent"]), 0.20)
        self.assertIs(p_ref_f["fused_mod"], True)

    def test_valid_route_and_profile_combinations_return_expected_profile_dict(self) -> None:
        self.assertTrue(
            hasattr(manifest_schema, "resolve_route_profile"),
            "manifest_schema must expose resolve_route_profile",
        )
        resolve_fn = manifest_schema.resolve_route_profile

        res_fl_fast = resolve_fn("fl2va", "fast")
        self.assertEqual(res_fl_fast["model_route"], "fl2va")
        self.assertEqual(res_fl_fast["profile"], "fast")

        res_ref_quality = resolve_fn("ref2va", "quality")
        self.assertEqual(res_ref_quality["model_route"], "ref2va")
        self.assertEqual(res_ref_quality["profile"], "quality")
        self.assertEqual(res_ref_quality["default_scheduler"], "beta")

        res_by_full_id = resolve_fn("ref2va", "ref2va_bf16_fast")
        self.assertEqual(res_by_full_id["model_route"], "ref2va")
        self.assertEqual(res_by_full_id["profile"], "fast")
        self.assertEqual(res_by_full_id["dense_blocks"], "0,1,48,49")

    def test_invalid_or_mismatched_route_profile_raises_value_error(self) -> None:
        self.assertTrue(
            hasattr(manifest_schema, "resolve_route_profile"),
            "manifest_schema must expose resolve_route_profile",
        )
        resolve_fn = manifest_schema.resolve_route_profile

        with self.assertRaises(ValueError):
            resolve_fn("unknown_route", "fast")

        with self.assertRaises(ValueError):
            resolve_fn("fl2va", "invalid_tier")

        with self.assertRaises(ValueError):
            resolve_fn("fl2va", "ref2va_bf16_fast")


class TestBatchManifestRouting(unittest.TestCase):
    def _write_json(self, tmp_dir: str, payload: dict) -> Path:
        manifest_path = Path(tmp_dir) / "manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return manifest_path

    def test_omitted_model_route_with_reference_assets_auto_infers_ref2va_and_profile_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_json(
                tmp,
                {
                    "batch_id": "ref_batch_q",
                    "title": "Ref2VA Quality Batch",
                    "profile": "quality",
                    "storyboards": [
                        {
                            "shot_id": "shot01",
                            "title": "Ref Shot",
                            "prompt": "A character speaking in studio lighting.",
                            "ref_images": ["assets/char1.png", "assets/char2.png"],
                            "ref_image_size": "match",
                        }
                    ],
                    "jobs": [
                        {"job_id": "job01", "shot_id": "shot01", "seed": 42},
                    ],
                },
            )
            manifest = manifest_schema.BatchManifest.from_json_file(manifest_path)
            self.assertEqual(manifest.model_route, "ref2va")
            self.assertEqual(manifest.profile, "quality")
            self.assertEqual(manifest.steps, 20)
            self.assertEqual(manifest.scheduler, "beta")

    def test_explicit_fl2va_fast_profile_applies_8_steps_and_simple_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_json(
                tmp,
                {
                    "batch_id": "fl_batch_fast",
                    "title": "FL2VA Fast Batch",
                    "model_route": "fl2va",
                    "profile": "fast",
                    "storyboards": [
                        {
                            "shot_id": "shot01",
                            "title": "FL2VA Shot",
                            "prompt": "Cinematic tracking shot across a neon street.",
                            "first_frame": "assets/first.png",
                            "last_frame": "assets/last.png",
                        }
                    ],
                    "jobs": [
                        {"job_id": "job01", "shot_id": "shot01", "seed": 101},
                    ],
                },
            )
            manifest = manifest_schema.BatchManifest.from_json_file(manifest_path)
            self.assertEqual(manifest.model_route, "fl2va")
            self.assertEqual(manifest.profile, "fast")
            self.assertEqual(manifest.steps, 8)
            self.assertEqual(manifest.scheduler, "simple")

    def test_ref2va_shot_without_any_reference_assets_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_json(
                tmp,
                {
                    "batch_id": "ref_empty",
                    "model_route": "ref2va",
                    "profile": "fast",
                    "storyboards": [
                        {
                            "shot_id": "shot01",
                            "prompt": "Prompt without reference assets.",
                        }
                    ],
                    "jobs": [{"job_id": "job01", "shot_id": "shot01", "seed": 1}],
                },
            )
            with self.assertRaises(ValueError):
                manifest_schema.BatchManifest.from_json_file(manifest_path)

    def test_ref2va_exceeding_asset_count_limits_or_invalid_ref_image_size_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p_too_many_imgs = self._write_json(
                tmp,
                {
                    "batch_id": "ref_over_img",
                    "model_route": "ref2va",
                    "storyboards": [
                        {
                            "shot_id": "shot01",
                            "prompt": "Too many ref images.",
                            "ref_images": [f"assets/img_{i}.png" for i in range(10)],
                        }
                    ],
                    "jobs": [{"job_id": "job01", "shot_id": "shot01", "seed": 1}],
                },
            )
            with self.assertRaises(ValueError):
                manifest_schema.BatchManifest.from_json_file(p_too_many_imgs)

            p_too_many_vids = self._write_json(
                tmp,
                {
                    "batch_id": "ref_over_vid",
                    "model_route": "ref2va",
                    "storyboards": [
                        {
                            "shot_id": "shot01",
                            "prompt": "Too many ref videos.",
                            "ref_videos": [f"assets/vid_{i}.mp4" for i in range(4)],
                        }
                    ],
                    "jobs": [{"job_id": "job01", "shot_id": "shot01", "seed": 1}],
                },
            )
            with self.assertRaises(ValueError):
                manifest_schema.BatchManifest.from_json_file(p_too_many_vids)

            p_too_many_auds = self._write_json(
                tmp,
                {
                    "batch_id": "ref_over_aud",
                    "model_route": "ref2va",
                    "storyboards": [
                        {
                            "shot_id": "shot01",
                            "prompt": "Too many ref audios.",
                            "ref_audios": [f"assets/aud_{i}.wav" for i in range(4)],
                        }
                    ],
                    "jobs": [{"job_id": "job01", "shot_id": "shot01", "seed": 1}],
                },
            )
            with self.assertRaises(ValueError):
                manifest_schema.BatchManifest.from_json_file(p_too_many_auds)

            p_invalid_size = self._write_json(
                tmp,
                {
                    "batch_id": "ref_bad_size",
                    "model_route": "ref2va",
                    "storyboards": [
                        {
                            "shot_id": "shot01",
                            "prompt": "Invalid ref_image_size.",
                            "ref_images": ["assets/img_0.png"],
                            "ref_image_size": "stretch_invalid",
                        }
                    ],
                    "jobs": [{"job_id": "job01", "shot_id": "shot01", "seed": 1}],
                },
            )
            with self.assertRaises(ValueError):
                manifest_schema.BatchManifest.from_json_file(p_invalid_size)

    def test_fl2va_route_with_reference_assets_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._write_json(
                tmp,
                {
                    "batch_id": "fl_with_refs",
                    "model_route": "fl2va",
                    "profile": "fast",
                    "storyboards": [
                        {
                            "shot_id": "shot01",
                            "prompt": "Invalid ref_images on fl2va route.",
                            "first_frame": "assets/first.png",
                            "ref_images": ["assets/ref1.png"],
                        }
                    ],
                    "jobs": [{"job_id": "job01", "shot_id": "shot01", "seed": 1}],
                },
            )
            with self.assertRaises(ValueError):
                manifest_schema.BatchManifest.from_json_file(manifest_path)


class TestLocalBatchPlanner(unittest.TestCase):
    def test_fl2va_and_ref2va_batches_pack_assets_and_deduplicate_conditioning_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets_dir = root / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            first_png = assets_dir / "first.png"
            last_png = assets_dir / "last.png"
            ref_a_png = assets_dir / "ref_a.png"
            ref_b_png = assets_dir / "ref_b.png"
            first_png.write_bytes(_make_valid_rgb_png_bytes(255, 0, 0))
            last_png.write_bytes(_make_valid_rgb_png_bytes(0, 255, 0))
            ref_a_png.write_bytes(_make_valid_rgb_png_bytes(0, 0, 255))
            ref_b_png.write_bytes(_make_valid_rgb_png_bytes(255, 255, 0))

            # 1) Test fl2va manifest packing
            fl_manifest_path = root / "fl2va_manifest.json"
            fl_tar_path = root / "fl2va_input_jobs.tar"
            fl_manifest_path.write_text(
                json.dumps(
                    {
                        "batch_id": "fl_pack_test",
                        "title": "FL2VA Pack Test",
                        "model_route": "fl2va",
                        "profile": "quality",
                        "storyboards": [
                            {
                                "shot_id": "shot_fl",
                                "title": "FL Shot",
                                "prompt": "Sunlight over mountain ridge.",
                                "first_frame": "assets/first.png",
                                "last_frame": "assets/last.png",
                            }
                        ],
                        "jobs": [
                            {"job_id": "j1", "shot_id": "shot_fl", "seed": 11},
                            {"job_id": "j2", "shot_id": "shot_fl", "seed": 22},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            fl_plan = local_planner.analyze_and_pack_batch(fl_manifest_path, fl_tar_path)
            self.assertTrue(fl_tar_path.exists())
            self.assertEqual(fl_plan["num_unique_conditionings"], 1)
            self.assertEqual(fl_plan["cond_dedup_hits"], 1)

            with tarfile.open(fl_tar_path, "r") as tf:
                names = {m.name.lstrip("./") for m in tf.getmembers()}
                self.assertIn("batch_manifest_resolved.json", names)
                resolved_member = next(
                    m for m in tf.getmembers() if m.name.lstrip("./") == "batch_manifest_resolved.json"
                )
                resolved_fl = json.loads(tf.extractfile(resolved_member).read().decode("utf-8"))
                self.assertEqual(resolved_fl["model_route"], "fl2va")
                self.assertEqual(resolved_fl["profile"], "quality")
                self.assertEqual(resolved_fl["profile_id"], "fl2va_bf16_quality")
                self.assertEqual(resolved_fl["scheduler"], "simple")
                self.assertIn("locked_inference_config", resolved_fl)

            # 2) Test ref2va manifest packing & conditioning hash sensitivity
            ref_manifest_path = root / "ref2va_manifest.json"
            ref_tar_path = root / "ref2va_input_jobs.tar"
            ref_manifest_path.write_text(
                json.dumps(
                    {
                        "batch_id": "ref_pack_test",
                        "title": "Ref2VA Pack Test",
                        "model_route": "ref2va",
                        "profile": "fast",
                        "storyboards": [
                            {
                                "shot_id": "shot_ref_1",
                                "title": "Ref Shot Match A",
                                "prompt": "Character walking through rain.",
                                "ref_images": ["assets/ref_a.png"],
                                "ref_image_size": "match",
                            },
                            {
                                "shot_id": "shot_ref_2",
                                "title": "Ref Shot Max A",
                                "prompt": "Character walking through rain.",
                                "ref_images": ["assets/ref_a.png"],
                                "ref_image_size": "max",
                            },
                            {
                                "shot_id": "shot_ref_3",
                                "title": "Ref Shot Match B",
                                "prompt": "Character walking through rain.",
                                "ref_images": ["assets/ref_b.png"],
                                "ref_image_size": "match",
                            },
                        ],
                        "jobs": [
                            {"job_id": "rj1", "shot_id": "shot_ref_1", "seed": 100},
                            {"job_id": "rj2", "shot_id": "shot_ref_1", "seed": 200},
                            {"job_id": "rj3", "shot_id": "shot_ref_2", "seed": 100},
                            {"job_id": "rj4", "shot_id": "shot_ref_3", "seed": 100},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            ref_plan = local_planner.analyze_and_pack_batch(ref_manifest_path, ref_tar_path)
            self.assertTrue(ref_tar_path.exists())
            self.assertEqual(ref_plan["num_unique_conditionings"], 3)
            self.assertEqual(ref_plan["cond_dedup_hits"], 1)

            with tarfile.open(ref_tar_path, "r") as tf:
                names = {m.name.lstrip("./") for m in tf.getmembers()}
                self.assertIn("batch_manifest_resolved.json", names)
                resolved_member = next(
                    m for m in tf.getmembers() if m.name.lstrip("./") == "batch_manifest_resolved.json"
                )
                resolved_ref = json.loads(tf.extractfile(resolved_member).read().decode("utf-8"))
                self.assertEqual(resolved_ref["model_route"], "ref2va")
                self.assertEqual(resolved_ref["profile"], "fast")
                self.assertEqual(resolved_ref["profile_id"], "ref2va_bf16_fast")
                self.assertEqual(resolved_ref["scheduler"], "simple")
                self.assertIn("locked_inference_config", resolved_ref)

                shots_by_id = {s["shot_id"]: s for s in resolved_ref["storyboards"]}
                h1 = shots_by_id["shot_ref_1"]["cond_hash"]
                h2 = shots_by_id["shot_ref_2"]["cond_hash"]
                h3 = shots_by_id["shot_ref_3"]["cond_hash"]
                self.assertNotEqual(h1, h2, "Different ref_image_size must yield distinct cond_hash")
                self.assertNotEqual(h1, h3, "Different ref_images content must yield distinct cond_hash")

                jobs_by_id = {j["job_id"]: j for j in resolved_ref["jobs"]}
                self.assertEqual(jobs_by_id["rj1"]["cond_hash"], jobs_by_id["rj2"]["cond_hash"])

                for s_entry in resolved_ref["storyboards"]:
                    for rel_img in s_entry.get("ref_images", []):
                        self.assertIn(rel_img.lstrip("./"), names)


if __name__ == "__main__":
    unittest.main()
