"""Provenance observation must not change model providers or inference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from retrieval.adapters.face_model import build_face_app, configure_face_providers, face_model_files, face_runtime
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

    def test_cpu_configuration_does_not_replace_sessions(self):
        session = mock.Mock()
        app = SimpleNamespace(models={"recognition": SimpleNamespace(session=session)})
        configure_face_providers(app, "cpu")
        session.set_providers.assert_not_called()

    def test_cuda_is_explicit_for_every_facial_model(self):
        sessions = [mock.Mock(), mock.Mock()]
        for session in sessions:
            session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = SimpleNamespace(models={name: SimpleNamespace(session=session)
                                     for name, session in zip(("detection", "recognition"), sessions)})
        ort = mock.Mock()
        ort.get_available_providers.return_value = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        with mock.patch.dict(sys.modules, {"onnxruntime": ort}):
            configure_face_providers(app, "cuda")
        for session in sessions:
            session.set_providers.assert_called_once_with(["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_unavailable_cuda_is_not_silent_cpu_fallback(self):
        ort = mock.Mock()
        ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with mock.patch.dict(sys.modules, {"onnxruntime": ort}), self.assertRaisesRegex(RuntimeError, "indisponível"):
            configure_face_providers(SimpleNamespace(models={}), "cuda")

    def test_failed_cuda_activation_is_reported(self):
        ort = mock.Mock()
        ort.get_available_providers.return_value = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        session = mock.Mock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        app = SimpleNamespace(models={"recognition": SimpleNamespace(session=session)})
        with mock.patch.dict(sys.modules, {"onnxruntime": ort}), self.assertRaisesRegex(RuntimeError, "não ativou CUDA"):
            configure_face_providers(app, "cuda")

    def test_build_configures_providers_before_preparing_models(self):
        app = mock.Mock()
        face_analysis = mock.Mock(return_value=app)
        with mock.patch.dict(sys.modules, {"torch": mock.Mock(), "insightface.app": SimpleNamespace(FaceAnalysis=face_analysis)}), \
                mock.patch("retrieval.adapters.face_model.configure_face_providers") as configure:
            configure.side_effect = lambda *_: app.record_configuration()
            self.assertIs(build_face_app("cuda", 640), app)
        self.assertEqual(app.mock_calls, [mock.call.record_configuration(), mock.call.prepare(ctx_id=0, det_size=(640, 640))])


if __name__ == "__main__":
    unittest.main()
