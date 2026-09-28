from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "environment" / "EX-007_environment_report.json"
PACKAGES = (
    "insightface",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "opencv-python",
    "numpy",
    "pandas",
    "tqdm",
    "matplotlib",
    "pillow",
    "pillow-heif",
    "scikit-learn",
    "streamlit",
    "kagglehub",
    "torch",
    "torchvision",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_state() -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"])
    status = command_output(["git", "status", "--porcelain"])
    return {
        "commit": commit.get("stdout") if commit.get("returncode") == 0 else None,
        "dirty": bool(status.get("stdout")) if status.get("returncode") == 0 else None,
    }


def torch_cuda_check() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depende do ambiente local
        return {"passed": False, "error": f"Falha ao importar torch: {exc}"}

    available = bool(torch.cuda.is_available())
    result: dict[str, Any] = {
        "passed": False,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": available,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if available else 0,
    }
    if not available:
        result["error"] = "torch.cuda.is_available() retornou False."
        return result

    try:
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        product = left @ left.T
        torch.cuda.synchronize(device)
        expected = left.cpu() @ left.cpu().T
        passed = bool(torch.equal(product.cpu(), expected))
        result.update(
            {
                "passed": passed,
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": int(properties.total_memory),
                "smoke_test": "matrix_multiplication",
            }
        )
        if not passed:
            result["error"] = "O resultado do calculo CUDA divergiu do calculo em CPU."
    except Exception as exc:  # pragma: no cover - depende do ambiente local
        result["error"] = f"Falha no teste CUDA do PyTorch: {exc}"
    return result


def onnx_cuda_check() -> dict[str, Any]:
    try:
        # Carregar PyTorch primeiro permite ao ONNX Runtime reutilizar DLLs CUDA/cuDNN.
        importlib.import_module("torch")
        ort = importlib.import_module("onnxruntime")
        onnx = importlib.import_module("onnx")
        helper = onnx.helper
        tensor_proto = onnx.TensorProto
    except Exception as exc:  # pragma: no cover - depende do ambiente local
        return {"passed": False, "error": f"Falha ao importar ONNX Runtime/ONNX: {exc}"}

    providers = list(ort.get_available_providers())
    result: dict[str, Any] = {
        "passed": False,
        "available_providers": providers,
    }
    if "CUDAExecutionProvider" not in providers:
        result["error"] = "CUDAExecutionProvider nao esta disponivel."
        return result

    try:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        input_info = helper.make_tensor_value_info("x", tensor_proto.FLOAT, [1, 4])
        output_info = helper.make_tensor_value_info("y", tensor_proto.FLOAT, [1, 4])
        one = helper.make_tensor("one", tensor_proto.FLOAT, [1], [1.0])
        node = helper.make_node("Add", ["x", "one"], ["y"])
        graph = helper.make_graph([node], "cuda_smoke_test", [input_info], [output_info], [one])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9
        session = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider"],
        )
        numpy = importlib.import_module("numpy")
        output = session.run(None, {"x": numpy.zeros((1, 4), dtype=numpy.float32)})[0]
        passed = bool(numpy.allclose(output, numpy.ones((1, 4), dtype=numpy.float32)))
        active_providers = list(session.get_providers())
        passed = passed and active_providers[0] == "CUDAExecutionProvider"
        result.update(
            {
                "passed": passed,
                "active_providers": active_providers,
                "smoke_test": "onnx_add",
            }
        )
        if not passed:
            result["error"] = "A sessao nao executou prioritariamente no CUDAExecutionProvider."
    except Exception as exc:  # pragma: no cover - depende do ambiente local
        result["error"] = f"Falha no teste CUDA do ONNX Runtime: {exc}"
    return result


def build_report() -> dict[str, Any]:
    torch_check = torch_cuda_check()
    onnx_check = onnx_cuda_check()
    return {
        "schema_version": "1.0",
        "execution_id": "EX-007",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_gpu_experiments": bool(torch_check["passed"] and onnx_check["passed"]),
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "packages": {name: package_version(name) for name in PACKAGES},
        },
        "hardware": {
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            )
        },
        "checks": {
            "pytorch_cuda": torch_check,
            "onnxruntime_cuda": onnx_check,
        },
        "git": git_state(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida o ambiente CUDA unico antes dos experimentos completos."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Retorna codigo diferente de zero se PyTorch ou ONNX Runtime nao usar CUDA.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    state = "APROVADO" if report["ready_for_gpu_experiments"] else "REPROVADO"
    print(f"EX-007: {state}")
    print(f"Relatorio: {output}")
    if args.require_cuda and not report["ready_for_gpu_experiments"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
