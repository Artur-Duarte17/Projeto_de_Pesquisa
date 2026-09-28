"""Provenance observation must not change model providers or inference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from retrieval.adapters.face_model import face_model_files, face_runtime
from retrieval.adapters.global_model import global_weights_file


class ModelProvenanceTests(unittest.TestCase):
    def test_actual_providers_are_observed_without_changing_sessions(self):
        session = mock.Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        app = SimpleNamespace(models={"recognition": SimpleNamespace(session=session)})
        self.assertEqual(face_runtime(app)["recognition"]["providers"], ["CPUExecutionProvider"])
        session.set_providers.assert_not_called()

    def test_model_inputs_use_actual_loaded_file_names(self):
        app = SimpleNamespace(models={
            "detection": SimpleNamespace(model_file="models/detection.onnx"),
            "recognition": SimpleNamespace(model_file=None),
        })
        self.assertEqual(face_model_files(app), {
            "detection_weights": Path("models/detection.onnx"), "recognition_weights": None})

    def test_untrained_global_configuration_does_not_claim_a_checkpoint(self):
        self.assertIsNone(global_weights_file("none"))


if __name__ == "__main__":
    unittest.main()
