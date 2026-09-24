from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval.adapters.files import (
    build_query_exclusions,
    load_inventory_image_ids,
    sha256_file,
)
from retrieval.adapters.global_model import extract_global_embeddings_batch
from retrieval.adapters.manifest import write_run_manifest
from retrieval.adapters.search_gateway import (
    search_face_index,
    search_fusion,
    search_global_index,
)
from retrieval.domain.fusion import classify_metric_delta, fuse_cosine_score_matrices
from retrieval.domain.metrics import (
    apply_relevance_exclusions,
    average_precision,
    precision_at_k,
    select_results_for_storage,
    summarize_rankings,
)


def load_script_module(module_name: str, relative_path: str):
    script_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load test target: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE_COLLECTION = load_script_module(
    "prepare_collection_for_tests",
    "scripts/prepare_collection.py",
)
FINAL_GALLAGHER = load_script_module(
    "run_final_gallagher_for_tests",
    "scripts/run_final_gallagher.py",
)
FINAL_VALIDATION = load_script_module(
    "run_final_validation_for_tests",
    "scripts/run_final_validation.py",
)


class InventoryImageIdTests(unittest.TestCase):
    def test_inventory_ids_are_used_for_every_indexed_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "images"
            input_dir.mkdir()
            first = input_dir / "first.jpg"
            second = input_dir / "second.jpg"
            inventory = root / "inventory.csv"
            inventory.write_text(
                "image_id,file_name\ncollection_first,first.jpg\ncollection_second,second.jpg\n",
                encoding="utf-8",
            )

            mapping = load_inventory_image_ids(inventory, input_dir, [first, second])

            self.assertEqual(mapping[first.resolve()], "collection_first")
            self.assertEqual(mapping[second.resolve()], "collection_second")

    def test_inventory_rejects_missing_indexed_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "images"
            input_dir.mkdir()
            inventory = root / "inventory.csv"
            inventory.write_text(
                "image_id,file_name\ncollection_first,first.jpg\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not map 1 indexed image"):
                load_inventory_image_ids(
                    inventory,
                    input_dir,
                    [input_dir / "first.jpg", input_dir / "missing.jpg"],
                )

    def test_inventory_rejects_duplicate_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "images"
            input_dir.mkdir()
            inventory = root / "inventory.csv"
            inventory.write_text(
                "image_id,file_name\nduplicate,first.jpg\nduplicate,second.jpg\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate image_id"):
                load_inventory_image_ids(
                    inventory,
                    input_dir,
                    [input_dir / "first.jpg", input_dir / "second.jpg"],
                )


class CollectionPreparationTests(unittest.TestCase):
    def test_commands_use_external_input_and_separate_indexes(self) -> None:
        commands = PREPARE_COLLECTION.build_commands(
            python_executable="python.exe",
            input_dir=Path(r"C:\Albums\Familia"),
            output_dir=Path(r"C:\Projeto\outputs\collections\familia"),
            device="cuda",
            det_size=640,
            batch_size=16,
            workers=2,
            max_images=139,
        )

        self.assertEqual(len(commands), 2)
        self.assertIn(r"C:\Albums\Familia", commands[0])
        self.assertIn(r"C:\Albums\Familia", commands[1])
        self.assertIn(r"C:\Projeto\outputs\collections\familia\face", commands[0])
        self.assertIn(r"C:\Projeto\outputs\collections\familia\global", commands[1])
        self.assertIn("--no-identity-from-parent", commands[0])
        self.assertIn("--no-label-from-parent", commands[1])
        self.assertEqual(commands[0][-2:], ["--max-images", "139"])
        self.assertEqual(commands[1][-2:], ["--max-images", "139"])

    def test_final_gallagher_commands_cover_the_complete_pipeline(self) -> None:
        commands = FINAL_GALLAGHER.build_commands(
            python_executable="python.exe",
            device="cuda",
            batch_size=32,
            workers=4,
            det_size=640,
            overwrite=False,
            include_environment_check=True,
            include_error_analysis=True,
        )

        scripts = [Path(command[1]).name for command in commands]
        self.assertEqual(
            scripts,
            [
                "validate_environment.py",
                "01_index_faces.py",
                "01_index_global_resnet.py",
                "04_prepare_gallagher_eval.py",
                "03_prepare_paired_gallagher.py",
                "04_evaluate_paired_gallagher.py",
                "05_analyze_paired_fusion_errors.py",
            ],
        )
        self.assertIn("--require-cuda", commands[0])
        self.assertTrue(all("--overwrite" not in command for command in commands))

    def test_complete_final_validation_covers_all_scientific_layers(self) -> None:
        commands = FINAL_VALIDATION.build_commands(
            python_executable="python.exe",
            device="cuda",
            batch_size=32,
            evaluation_batch_size=128,
            workers=4,
            det_size=640,
            overwrite=False,
            allow_dirty=False,
            include_gallagher_error_analysis=True,
        )

        scripts = [Path(command[1]).name for command in commands]
        self.assertEqual(
            scripts,
            [
                "validate_environment.py",
                "01_index_faces.py",
                "06_prepare_lfw_eval.py",
                "07_evaluate_lfw.py",
                "01_index_global_resnet.py",
                "04_evaluate_holidays.py",
                "run_final_gallagher.py",
            ],
        )
        self.assertIn("--skip-environment-check", commands[-1])
        self.assertNotIn("--allow-dirty", commands[-1])
        self.assertTrue(all("--overwrite" not in command for command in commands))


class IsolatedFinalRunTests(unittest.TestCase):
    @staticmethod
    def option(command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    def test_default_run_paths_keep_the_existing_layout(self) -> None:
        validation = FINAL_VALIDATION.paths_for_run()
        gallagher = FINAL_GALLAGHER.paths_for_run()

        self.assertEqual(validation.root, ROOT / "outputs" / "final")
        self.assertEqual(validation.lfw_protocol, ROOT / "data" / "evaluation" / "lfw_final")
        self.assertEqual(gallagher.root, validation.root)
        self.assertEqual(gallagher.final, validation.root / "gallagher")
        self.assertEqual(
            gallagher.evaluation, ROOT / "data" / "evaluation" / "gallagher_final"
        )
        self.assertEqual(gallagher.query, ROOT / "data" / "query" / "gallagher_final")

    def test_custom_run_root_is_shared_and_redirects_every_generated_path(self) -> None:
        requested = Path("outputs/validation_runs/architecture_candidate")
        root = (ROOT / requested).resolve()
        validation = FINAL_VALIDATION.paths_for_run(requested)
        gallagher = FINAL_GALLAGHER.paths_for_run(requested)

        self.assertEqual(validation.root, root)
        self.assertEqual(validation.lfw_protocol, root / "protocols" / "lfw")
        self.assertEqual(gallagher.root, root)
        self.assertEqual(gallagher.final, root / "gallagher")
        self.assertEqual(gallagher.evaluation, root / "protocols" / "gallagher")
        self.assertEqual(gallagher.query, root / "queries" / "gallagher")
        self.assertEqual(gallagher.environment, root / "environment_report.json")
        validation_protected = FINAL_VALIDATION.protected_outputs(requested)
        gallagher_protected = FINAL_GALLAGHER.protected_outputs(requested)
        self.assertIn(root / "final_validation_manifest.json", validation_protected)
        self.assertIn(root / "gallagher" / "final_pipeline_manifest.json", gallagher_protected)
        for protected in (*validation_protected, *gallagher_protected):
            self.assertTrue(protected.is_relative_to(root))

        validation_commands = FINAL_VALIDATION.build_commands(
            python_executable="python.exe",
            device="cuda",
            batch_size=32,
            evaluation_batch_size=128,
            workers=4,
            det_size=640,
            overwrite=False,
            allow_dirty=False,
            include_gallagher_error_analysis=True,
            run_root=requested,
        )
        self.assertEqual(len(validation_commands), 7)
        self.assertEqual(
            self.option(validation_commands[0], "--output"),
            str(root / "environment_report.json"),
        )
        self.assertEqual(
            self.option(validation_commands[1], "--output-dir"),
            str(root / "lfw" / "face_index"),
        )
        self.assertEqual(
            self.option(validation_commands[2], "--output-dir"),
            str(root / "protocols" / "lfw"),
        )
        self.assertEqual(
            self.option(validation_commands[2], "--manifest-dir"),
            str(root / "lfw" / "protocol"),
        )
        self.assertEqual(
            self.option(validation_commands[3], "--queries-csv"),
            str(root / "protocols" / "lfw" / "lfw_face_queries.csv"),
        )
        self.assertEqual(
            self.option(validation_commands[3], "--relevance-csv"),
            str(root / "protocols" / "lfw" / "lfw_face_relevance.csv"),
        )
        self.assertEqual(
            self.option(validation_commands[3], "--output-dir"),
            str(root / "lfw" / "evaluation"),
        )
        self.assertEqual(
            self.option(validation_commands[4], "--output-dir"),
            str(root / "holidays" / "global_index"),
        )
        self.assertEqual(
            self.option(validation_commands[5], "--output-dir"),
            str(root / "holidays" / "evaluation"),
        )
        self.assertEqual(self.option(validation_commands[6], "--run-root"), str(root))
        self.assertIn("--skip-environment-check", validation_commands[6])

        gallagher_commands = FINAL_GALLAGHER.build_commands(
            python_executable="python.exe",
            device="cuda",
            batch_size=32,
            workers=4,
            det_size=640,
            overwrite=False,
            include_environment_check=True,
            include_error_analysis=True,
            run_root=requested,
        )
        self.assertEqual(len(gallagher_commands), 7)
        self.assertEqual(
            self.option(gallagher_commands[0], "--output"),
            str(root / "environment_report.json"),
        )
        self.assertEqual(
            self.option(gallagher_commands[1], "--output-dir"),
            str(root / "gallagher" / "face_index"),
        )
        self.assertEqual(
            self.option(gallagher_commands[2], "--output-dir"),
            str(root / "gallagher" / "global_index"),
        )
        self.assertEqual(
            self.option(gallagher_commands[3], "--output-dir"),
            str(root / "protocols" / "gallagher"),
        )
        self.assertEqual(
            self.option(gallagher_commands[3], "--query-crop-dir"),
            str(root / "queries" / "gallagher"),
        )
        self.assertEqual(
            self.option(gallagher_commands[3], "--manifest-dir"),
            str(root / "gallagher" / "face_protocol"),
        )
        self.assertEqual(
            self.option(gallagher_commands[4], "--output-csv"),
            str(root / "protocols" / "gallagher" / "gallagher_fusion_queries.csv"),
        )
        self.assertEqual(
            self.option(gallagher_commands[4], "--query-manifest-csv"),
            str(root / "protocols" / "gallagher" / "gallagher_fusion_query_manifest.csv"),
        )
        self.assertEqual(
            self.option(gallagher_commands[5], "--output-dir"),
            str(root / "gallagher" / "evaluation"),
        )
        self.assertEqual(
            self.option(gallagher_commands[6], "--output-dir"),
            str(root / "gallagher" / "error_analysis"),
        )

        generated_flags = (
            "--output", "--output-dir", "--output-csv", "--query-manifest-csv",
            "--query-crop-dir", "--manifest-dir",
        )
        for command in (*validation_commands, *gallagher_commands):
            self.assertNotIn("--overwrite", command)
            for flag in generated_flags:
                if flag in command:
                    self.assertTrue(
                        Path(self.option(command, flag)).is_relative_to(root),
                        f"{Path(command[1]).name} sends {flag} outside the isolated run",
                    )

    def test_run_root_rejects_baseline_and_unsafe_locations(self) -> None:
        unsafe_roots = (
            ROOT,
            ROOT / "data",
            Path("outputs"),
            Path("outputs/final"),
            Path("outputs/final/accidental_child"),
        )
        for unsafe in unsafe_roots:
            with self.subTest(root=unsafe):
                with self.assertRaises(ValueError):
                    FINAL_VALIDATION.paths_for_run(unsafe)
                with self.assertRaises(ValueError):
                    FINAL_GALLAGHER.paths_for_run(unsafe)

    def test_nonempty_target_is_rejected_before_revalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "candidate"
            FINAL_VALIDATION.require_empty_target(target)
            target.mkdir()
            FINAL_GALLAGHER.require_empty_target(target)
            (target / "previous_partial_run.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                FINAL_VALIDATION.require_empty_target(target)
            with self.assertRaises(FileExistsError):
                FINAL_GALLAGHER.require_empty_target(target)


class MetricTests(unittest.TestCase):
    def test_precision_at_k_uses_requested_k_as_denominator(self) -> None:
        self.assertAlmostEqual(precision_at_k(["relevant"], {"relevant"}, 5), 0.2)

    def test_average_precision_uses_full_relevance_denominator(self) -> None:
        ranked = ["irrelevant", "a", "b"]
        expected = ((1 / 2) + (2 / 3)) / 3
        self.assertAlmostEqual(average_precision(ranked, {"a", "b", "missing"}), expected)

    def test_summary_records_integral_ranking_size(self) -> None:
        summary = summarize_rankings({"q1": ["a", "b", "c"]}, {"q1": {"a"}}, ks=(2,))
        self.assertEqual(int(summary.loc[0, "ranking_size"]), 3)
        self.assertAlmostEqual(float(summary.loc[0, "precision_at_2"]), 0.5)

    def test_storage_limit_does_not_modify_integral_ranking(self) -> None:
        ranking = pd.DataFrame({"image_id": ["a", "b", "c"]})
        saved = select_results_for_storage(ranking, 2)
        self.assertEqual(ranking["image_id"].tolist(), ["a", "b", "c"])
        self.assertEqual(saved["image_id"].tolist(), ["a", "b"])


class GlobalBatchExtractionTests(unittest.TestCase):
    def test_batch_preserves_order_and_normalizes_descriptors(self) -> None:
        tensors = [
            torch.tensor([3.0, 4.0], dtype=torch.float32),
            torch.tensor([0.0, 2.0], dtype=torch.float32),
        ]
        descriptors = extract_global_embeddings_batch(
            tensors,
            torch.nn.Identity(),
            torch.device("cpu"),
        )
        expected = np.asarray([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(descriptors, expected, rtol=1e-6, atol=1e-7)

    def test_empty_batch_returns_empty_float32_matrix(self) -> None:
        descriptors = extract_global_embeddings_batch(
            [],
            torch.nn.Identity(),
            torch.device("cpu"),
        )
        self.assertEqual(descriptors.shape, (0, 0))
        self.assertEqual(descriptors.dtype, np.float32)


class ExclusionTests(unittest.TestCase):
    def test_query_exclusions_apply_to_explicit_relevance(self) -> None:
        queries = pd.DataFrame(
            [
                {
                    "query_id": "q1",
                    "query_path": "missing.jpg",
                    "source_image_id": "source",
                    "exclude_image_ids": "blocked_a;blocked_b",
                }
            ]
        )
        metadata = pd.DataFrame(columns=["image_id", "image_path"])
        exclusions = build_query_exclusions(queries, metadata)
        cleaned = apply_relevance_exclusions(
            {"q1": {"source", "blocked_a", "kept"}},
            exclusions,
        )
        self.assertEqual(exclusions["q1"], {"source", "blocked_a", "blocked_b"})
        self.assertEqual(cleaned["q1"], {"kept"})

    def test_global_search_excludes_uploaded_copy_by_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            source = folder / "source.jpg"
            uploaded_copy = folder / "uploaded.jpg"
            other = folder / "other.jpg"
            source.write_bytes(b"same image bytes")
            uploaded_copy.write_bytes(b"same image bytes")
            other.write_bytes(b"different image bytes")

            metadata = pd.DataFrame(
                [
                    {"image_id": "source", "image_path": str(source), "sha256": sha256_file(source)},
                    {"image_id": "other", "image_path": str(other), "sha256": sha256_file(other)},
                ]
            )
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            result = search_global_index(
                np.asarray([1.0, 0.0], dtype=np.float32),
                embeddings,
                metadata,
                topk=None,
                query_path=uploaded_copy,
            )
            self.assertEqual(result["image_id"].tolist(), ["other"])

    def test_face_search_excludes_all_faces_from_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            source = folder / "source.jpg"
            other = folder / "other.jpg"
            source.write_bytes(b"source")
            other.write_bytes(b"other")
            metadata = pd.DataFrame(
                [
                    self._face_row("source_f00", "source", source),
                    self._face_row("source_f01", "source", source),
                    self._face_row("other_f00", "other", other),
                ]
            )
            embeddings = np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                dtype=np.float32,
            )
            result = search_face_index(
                np.asarray([1.0, 0.0], dtype=np.float32),
                embeddings,
                metadata,
                topk=None,
                query_path=source,
            )
            self.assertEqual(result["image_id"].tolist(), ["other"])

    def test_fusion_propagates_explicit_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            query = folder / "query.jpg"
            source = folder / "source.jpg"
            other = folder / "other.jpg"
            for path, content in ((query, b"query"), (source, b"source"), (other, b"other")):
                path.write_bytes(content)

            face_meta = pd.DataFrame(
                [
                    self._face_row("source_f00", "source", source),
                    self._face_row("other_f00", "other", other),
                ]
            )
            global_meta = pd.DataFrame(
                [
                    {"image_id": "source", "image_path": str(source)},
                    {"image_id": "other", "image_path": str(other)},
                ]
            )
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            result = search_fusion(
                query,
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([1.0, 0.0], dtype=np.float32),
                embeddings,
                face_meta,
                embeddings,
                global_meta,
                topk=None,
                face_weight=0.7,
                global_weight=0.3,
                exclude_image_ids={"source"},
            )
            self.assertEqual(result["image_id"].tolist(), ["other"])

    @staticmethod
    def _face_row(face_id: str, image_id: str, image_path: Path) -> dict[str, object]:
        return {
            "face_id": face_id,
            "image_id": image_id,
            "image_path": str(image_path),
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 10,
            "bbox_y2": 10,
        }


class FusionScoreTests(unittest.TestCase):
    def test_metric_delta_classification_respects_tolerance(self) -> None:
        self.assertEqual(classify_metric_delta(0.01), "improved")
        self.assertEqual(classify_metric_delta(-0.01), "degraded")
        self.assertEqual(classify_metric_delta(1e-13), "unchanged")

    def test_fixed_weight_fusion_uses_unit_cosines_and_missing_face_zero(self) -> None:
        face = np.asarray([[1.0, -0.5]], dtype=np.float32)
        global_scores = np.asarray([[0.0, 1.0]], dtype=np.float32)
        available = np.asarray([[True, False]])
        fused, face_unit, global_unit = fuse_cosine_score_matrices(
            face,
            global_scores,
            available,
            face_weight=0.7,
            global_weight=0.3,
        )
        np.testing.assert_allclose(face_unit, [[1.0, 0.0]], atol=1e-7)
        np.testing.assert_allclose(global_unit, [[0.5, 1.0]], atol=1e-7)
        np.testing.assert_allclose(fused, [[0.85, 0.3]], atol=1e-7)

    def test_fusion_weights_must_sum_to_one(self) -> None:
        scores = np.zeros((1, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            fuse_cosine_score_matrices(scores, scores, np.ones((1, 1), dtype=bool), 0.7, 0.4)


class ManifestTests(unittest.TestCase):
    def test_manifest_records_hashes_and_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            input_path = folder / "queries.csv"
            result_path = folder / "metrics.csv"
            manifest_path = folder / "run_manifest.json"
            input_path.write_text("query_id\nq1\n", encoding="utf-8")
            result_path.write_text("mAP\n1.0\n", encoding="utf-8")

            write_run_manifest(
                manifest_path,
                script_path=Path(__file__),
                method="synthetic",
                configuration={"save_topk": 1, "ranking_depth": "all_eligible_images"},
                inputs={"queries": input_path},
                result_files={"metrics": result_path},
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["method"], "synthetic")
            self.assertEqual(manifest["inputs"]["queries"]["sha256"], sha256_file(input_path))
            self.assertEqual(manifest["results"]["metrics"]["sha256"], sha256_file(result_path))


if __name__ == "__main__":
    unittest.main()
