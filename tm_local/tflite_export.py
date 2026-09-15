from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .utils import sha256_file, utc_now_iso

LOGGER = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]
VALID_TFLITE_FORMATS = ("int8", "uint8", "float32", "dynamic")


class ExportError(RuntimeError):
    pass


def representative_samples_sha256(samples: Sequence[np.ndarray]) -> str:
    """Digest the exact float32 tensors used for integer calibration.

    Shapes are included so byte-identical buffers with different tensor geometry cannot
    share conversion provenance.  The digest is stored in every integer model report and
    is required for cache reuse.
    """

    digest = hashlib.sha256()
    digest.update(f"tm-local-tflite-representative/v1/{len(samples)}".encode("ascii"))
    for sample in samples:
        array = np.ascontiguousarray(np.asarray(sample, dtype=np.float32))
        shape = ",".join(str(int(value)) for value in array.shape)
        digest.update(f"|{array.ndim}:{shape}|".encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _emit(progress: ProgressFn | None, value: float, message: str) -> None:
    if progress:
        progress(float(max(0.0, min(1.0, value))), str(message))


def _representative_generator(samples: Sequence[np.ndarray]):
    def generator():
        for sample in samples:
            array = np.asarray(sample, dtype=np.float32)
            if array.ndim < 1:
                raise ExportError("Representative sample has no dimensions.")
            yield [array[np.newaxis, ...]]

    return generator


def _single_input_shape(model: Any) -> tuple[int | None, ...]:
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ExportError("Only single-input classifiers are supported.")
        shape = shape[0]
    if not shape or len(shape) < 2:
        raise ExportError(f"Cannot determine the model input shape: {shape!r}")
    result: list[int | None] = [None]
    for value in shape[1:]:
        result.append(None if value is None else int(value))
    return tuple(result)


def export_inference_saved_model(model: Any, output_dir: Path) -> Path:
    """Export a stable inference-only SavedModel for TensorFlow Lite.

    TensorFlow 2.21 with legacy TF-Keras can occasionally spend an unbounded
    amount of time inside ``TFLiteConverter.from_keras_model``.  Saving one
    explicit inference signature and converting from SavedModel avoids the
    fragile in-memory Keras conversion path and also makes the conversion
    process independently restartable.
    """

    import tensorflow as tf

    destination = Path(output_dir)
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_shape = _single_input_shape(model)
    input_name = "inputs"
    try:
        raw_name = str(model.inputs[0].name).split(":", 1)[0]
        if raw_name:
            input_name = raw_name.replace("/", "_")
    except Exception:
        pass

    @tf.function(input_signature=[tf.TensorSpec(input_shape, tf.float32, name=input_name)])
    def serving(inputs):  # noqa: ANN001
        outputs = model(inputs, training=False)
        if isinstance(outputs, dict):
            return outputs
        if isinstance(outputs, (tuple, list)):
            return {f"output_{index}": value for index, value in enumerate(outputs)}
        return {"scores": outputs}

    concrete = serving.get_concrete_function()
    tf.saved_model.save(model, str(destination), signatures={"serving_default": concrete})
    if not (destination / "saved_model.pb").is_file():
        raise ExportError("SavedModel export did not create saved_model.pb.")
    return destination


def _converter_from_saved_model(saved_model_dir: Path):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_dir), signature_keys=["serving_default"]
    )
    # This is harmless for models without resource variables and helps legacy
    # TF-Keras models on current TensorFlow releases.
    converter.experimental_enable_resource_variables = True
    return converter


def convert_float32(saved_model_dir: Path) -> bytes:
    converter = _converter_from_saved_model(saved_model_dir)
    return converter.convert()


def convert_dynamic(saved_model_dir: Path) -> bytes:
    import tensorflow as tf

    converter = _converter_from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_full_integer(
    saved_model_dir: Path,
    samples: Sequence[np.ndarray],
    io_dtype: str,
) -> bytes:
    import tensorflow as tf

    if not samples:
        raise ExportError("Full integer quantization requires representative samples.")
    if io_dtype not in {"int8", "uint8"}:
        raise ExportError("io_dtype must be int8 or uint8.")
    converter = _converter_from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_generator(samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8 if io_dtype == "int8" else tf.uint8
    converter.inference_output_type = tf.int8 if io_dtype == "int8" else tf.uint8
    return converter.convert()


def _interpreter(model_path: Path, *, delegates: bool):
    """Build a TFLite interpreter, optionally without TensorFlow's default delegates."""

    import tensorflow as tf

    if delegates:
        return tf.lite.Interpreter(model_path=str(model_path))
    return tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )


def _allocated_interpreter(model_path: Path):
    """Allocate an interpreter, degrading to the reference kernels if XNNPACK cannot start.

    XNNPACK needs a runtime workspace, and that allocation is the one part of running a model
    that can fail for reasons that have nothing to do with the model: in a process that has been
    alive a long time -- the Studio server serving Preview after Preview, or a 582-test suite in
    one interpreter -- it raises "failed to create XNNPACK runtime". Falling back to the builtin
    kernels turns that into a slower prediction instead of a failed one.

    The fast path is untouched on purpose: when XNNPACK starts, it runs, so the numbers this
    returns -- and the top-1 agreement computed from them -- are exactly what they were before.
    Only the already-broken case behaves differently.
    """

    try:
        interpreter = _interpreter(model_path, delegates=True)
        interpreter.allocate_tensors()
        return interpreter
    except RuntimeError as exc:
        if "XNNPACK" not in str(exc):
            raise
        LOGGER.warning(
            "XNNPACK could not start for %s (%s); falling back to the builtin kernels",
            model_path.name,
            exc,
        )
    interpreter = _interpreter(model_path, delegates=False)
    interpreter.allocate_tensors()
    return interpreter


def inspect_tflite(model_path: Path) -> dict[str, Any]:
    # Audit the FILE, not a runtime's rewritten view of it, and never load the XNNPACK delegate.
    # Two reasons, both measured:
    #   1. Correctness. XNNPACK rewrites the graph at allocate_tensors(): it folds nodes into one
    #      synthetic DELEGATE op and hides the tensors it absorbed. On kws3_int8.tflite the
    #      delegated view reported 10 int8 + 3 int32 tensors and an operator list ending in
    #      DELEGATE; the undelegated view reports 12 int8 + 6 int32 and the model's real ops. The
    #      strict-integer gate asks whether this FILE has float tensors or Flex ops, so it must
    #      see every tensor the file declares. Fewer hidden tensors can only make the gate
    #      stricter -- strict_full_integer and float_tensors were identical in both views.
    #   2. Robustness. The delegate allocates an XNNPACK runtime workspace, and that allocation
    #      is the one part of an audit that can fail for reasons that have nothing to do with the
    #      model: late in a long-lived process it raised "failed to create XNNPACK runtime" and
    #      turned a passing export into an ExportError. This function never calls invoke(), so
    #      the delegate buys the audit nothing to begin with.
    interpreter = _interpreter(model_path, delegates=False)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    tensors = interpreter.get_tensor_details()
    try:
        ops = interpreter._get_ops_details()  # type: ignore[attr-defined]
    except Exception:
        ops = []

    def tensor_summary(detail: dict[str, Any]) -> dict[str, Any]:
        params = detail.get("quantization_parameters", {})
        scales = np.asarray(params.get("scales", []))
        zero_points = np.asarray(params.get("zero_points", []))
        quant = detail.get("quantization", (0.0, 0))
        return {
            "name": str(detail.get("name", "")),
            "shape": [int(v) for v in np.asarray(detail.get("shape", [])).tolist()],
            "shape_signature": [
                int(v)
                for v in np.asarray(
                    detail.get("shape_signature", detail.get("shape", []))
                ).tolist()
            ],
            "dtype": np.dtype(detail["dtype"]).name,
            "scale": float(quant[0]) if quant else 0.0,
            "zero_point": int(quant[1]) if quant else 0,
            "per_axis_scale_count": int(scales.size),
            "per_axis_zero_point_count": int(zero_points.size),
            "quantized_dimension": int(params.get("quantized_dimension", 0)),
        }

    dtype_counts: dict[str, int] = {}
    float_tensors: list[str] = []
    for detail in tensors:
        dtype = np.dtype(detail["dtype"]).name
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        if np.issubdtype(detail["dtype"], np.floating):
            float_tensors.append(str(detail.get("name", "")))
    operator_names = [str(item.get("op_name", "")) for item in ops]
    flex_ops = [name for name in operator_names if name.startswith("Flex")]
    custom_ops = [
        name for name in operator_names if name == "CUSTOM" or name.startswith("CUSTOM:")
    ]
    input_dtypes = [np.dtype(item["dtype"]).name for item in inputs]
    output_dtypes = [np.dtype(item["dtype"]).name for item in outputs]
    integer_io = all(dtype in {"int8", "uint8"} for dtype in input_dtypes + output_dtypes)
    return {
        "path": model_path.name,
        "size_bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
        "inputs": [tensor_summary(item) for item in inputs],
        "outputs": [tensor_summary(item) for item in outputs],
        "tensor_dtype_counts": dtype_counts,
        "float_tensor_count": len(float_tensors),
        "float_tensors": float_tensors[:100],
        "operators": operator_names,
        "flex_operators": flex_ops,
        "custom_operators": custom_ops,
        "requires_select_tf_ops": bool(flex_ops),
        "integer_input_output": integer_io,
        "strict_full_integer": integer_io and not float_tensors and not flex_ops and not custom_ops,
    }


def validate_tflite_format_report(report: dict[str, Any], model_format: str) -> None:
    """Reject a stale/mislabeled artifact before it can be reused or packaged."""

    name = str(model_format).strip().lower()
    if name not in VALID_TFLITE_FORMATS:
        raise ExportError(f"Unsupported TensorFlow Lite format: {name!r}.")
    inputs = report.get("inputs")
    outputs = report.get("outputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ExportError(f"{name} model must expose exactly one input tensor.")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ExportError(f"{name} model must expose exactly one output tensor.")
    input_dtype = str(inputs[0].get("dtype", ""))
    output_dtype = str(outputs[0].get("dtype", ""))
    if report.get("flex_operators") or report.get("custom_operators"):
        raise ExportError(f"{name} model contains Flex/custom operators.")
    if name in {"int8", "uint8"}:
        if input_dtype != name or output_dtype != name:
            raise ExportError(
                f"{name.upper()} model has {input_dtype}/{output_dtype} I/O instead of "
                f"{name}/{name}."
            )
        if not bool(report.get("strict_full_integer")):
            raise ExportError(
                f"{name.upper()} model contains Float/Flex/custom tensors or operators."
            )
    elif input_dtype != "float32" or output_dtype != "float32":
        raise ExportError(
            f"{name} model has {input_dtype}/{output_dtype} I/O instead of float32/float32."
        )


def _quantize(array: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return array.astype(dtype)
    scale, zero = detail.get("quantization", (0.0, 0))
    if not scale:
        raise ExportError("Integer TFLite input has no quantization scale.")
    info = np.iinfo(dtype)
    value = np.round(array / scale + zero)
    return np.clip(value, info.min, info.max).astype(dtype)


def _dequantize(array: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(detail["dtype"], np.floating):
        return array.astype(np.float32)
    scale, zero = detail.get("quantization", (0.0, 0))
    return (array.astype(np.float32) - float(zero)) * float(scale)


def predict_tflite(model_path: Path, batch: np.ndarray) -> np.ndarray:
    interpreter = _allocated_interpreter(model_path)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    results: list[np.ndarray] = []
    for item in np.asarray(batch):
        value = item[np.newaxis, ...].astype(np.float32)
        interpreter.set_tensor(input_detail["index"], _quantize(value, input_detail))
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        results.append(_dequantize(output, output_detail)[0])
    return np.stack(results, axis=0)


def compare_keras_and_tflite(
    model: Any,
    model_path: Path,
    samples: Sequence[np.ndarray],
) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    batch = np.stack([np.asarray(sample, dtype=np.float32) for sample in samples], axis=0)
    keras_output = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
    tflite_output = predict_tflite(model_path, batch)
    difference = np.abs(keras_output - tflite_output)
    keras_flat = keras_output.reshape(keras_output.shape[0], -1).astype(np.float64)
    tflite_flat = tflite_output.reshape(tflite_output.shape[0], -1).astype(np.float64)
    cosine = np.sum(keras_flat * tflite_flat, axis=1) / np.maximum(
        np.linalg.norm(keras_flat, axis=1) * np.linalg.norm(tflite_flat, axis=1),
        1e-12,
    )
    relative_l2 = np.linalg.norm(keras_flat - tflite_flat, axis=1) / np.maximum(
        np.linalg.norm(keras_flat, axis=1),
        1e-12,
    )
    return {
        "sample_count": int(batch.shape[0]),
        "top1_agreement": float(
            np.mean(np.argmax(keras_output, axis=1) == np.argmax(tflite_output, axis=1))
        ),
        # Does the per-output decision survive quantisation? top-1 agreement is blind to
        # this for a multi-label head, where several outputs can be "on" at once and the
        # argmax can be unchanged while individual decisions flip.
        "detection_agreement": float(
            np.mean((keras_output >= 0.5) == (tflite_output >= 0.5))
        ),
        "mean_absolute_error": float(np.mean(difference)),
        "maximum_absolute_error": float(np.max(difference)),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "minimum_cosine_similarity": float(np.min(cosine)),
        "mean_relative_l2_error": float(np.mean(relative_l2)),
    }


def convert_selected(
    model: Any,
    representative_samples: Sequence[np.ndarray],
    output_dir: Path,
    prefix: str,
    selected: Iterable[str],
    *,
    progress: ProgressFn | None = None,
    comparison_samples: int = 20,
    reuse_existing: bool = True,
    source_model_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Generate only requested formats from one inference-only SavedModel.

    TensorFlow Lite's Python converter does not expose fine-grained progress
    while ``convert()`` is running.  The caller therefore receives clear stage
    messages and elapsed time in the UI, while training itself remains a
    separate job.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_set = {str(item).strip().lower() for item in selected}
    unknown = sorted(selected_set.difference(VALID_TFLITE_FORMATS))
    if unknown:
        raise ExportError("Unsupported TensorFlow Lite format(s): " + ", ".join(unknown))
    ordered = [name for name in VALID_TFLITE_FORMATS if name in selected_set]
    if not ordered:
        return {}, {"generated_at": utc_now_iso(), "selected_formats": [], "models": {}}
    integer_requested = any(name in {"int8", "uint8"} for name in ordered)
    if integer_requested and len(representative_samples) < 1:
        raise ExportError("Integer quantization requires representative samples.")
    representative_sha256 = (
        representative_samples_sha256(representative_samples)
        if integer_requested
        else None
    )

    report_path = output_dir / "conversion_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    except Exception:
        report = {}
    report["generated_at"] = utc_now_iso()
    report["selected_formats"] = ordered
    report.setdefault("models", {})
    artifacts: dict[str, str] = {}

    labels = {
        "float32": "Converting Float32 TensorFlow Lite model",
        "dynamic": "Converting dynamic-range TensorFlow Lite model",
        "int8": "Calibrating and converting strict INT8 model",
        "uint8": "Calibrating and converting strict UINT8 model",
    }

    missing: list[str] = []
    for model_format in ordered:
        model_path = output_dir / f"{prefix}_{model_format}.tflite"
        if reuse_existing and model_path.is_file() and model_path.stat().st_size > 16:
            prior_report = report["models"].get(model_format)
            try:
                if source_model_sha256 is not None:
                    if not isinstance(prior_report, dict):
                        raise ExportError("Cached model has no conversion provenance.")
                    if prior_report.get("source_model_sha256") != source_model_sha256:
                        raise ExportError("Cached model belongs to a different Keras artifact.")
                if model_format in {"int8", "uint8"}:
                    if not isinstance(prior_report, dict):
                        raise ExportError("Cached integer model has no calibration provenance.")
                    if prior_report.get("representative_sha256") != representative_sha256:
                        raise ExportError(
                            "Cached integer model used different representative samples."
                        )
                existing_report = inspect_tflite(model_path)
                validate_tflite_format_report(existing_report, model_format)
                if (
                    isinstance(prior_report, dict)
                    and prior_report.get("sha256")
                    and prior_report.get("sha256") != existing_report.get("sha256")
                ):
                    raise ExportError("Cached model bytes changed after conversion.")
            except Exception:
                report["models"].pop(model_format, None)
                missing.append(model_format)
            else:
                artifacts[model_format] = model_path.name
                existing_report["reused_existing_file"] = True
                if source_model_sha256 is not None:
                    existing_report["source_model_sha256"] = source_model_sha256
                # A cached integer model is still evaluated against the current
                # representative inputs.  Shape/operator inspection alone proves
                # that the file is a valid integer graph, but not that its 1024-D
                # embedding remains numerically faithful to this Keras artifact.
                if model_format in {"int8", "uint8"}:
                    existing_report["representative_sha256"] = representative_sha256
                    existing_report["comparison"] = compare_keras_and_tflite(
                        model,
                        model_path,
                        representative_samples[:comparison_samples],
                    )
                report["models"][model_format] = existing_report
        else:
            missing.append(model_format)

    if missing:
        with tempfile.TemporaryDirectory(
            prefix="tm_local_savedmodel_", dir=output_dir.parent
        ) as temp:
            saved_model_dir = Path(temp) / "inference_saved_model"
            _emit(progress, 0.01, "Exporting a stable inference graph…")
            export_started = time.perf_counter()
            export_inference_saved_model(model, saved_model_dir)
            report["saved_model_export_seconds"] = float(
                time.perf_counter() - export_started
            )
            _emit(
                progress,
                0.10,
                "Inference graph exported. The selected converter is now running; "
                "its percentage may remain unchanged until this format finishes…",
            )

            total = len(missing)
            for index, model_format in enumerate(missing):
                stage_start = 0.10 + 0.86 * index / total
                stage_end = 0.10 + 0.86 * (index + 1) / total
                _emit(
                    progress,
                    stage_start,
                    labels[model_format]
                    + "… TensorFlow Lite does not report sub-progress for this step.",
                )
                started = time.perf_counter()
                model_path = output_dir / f"{prefix}_{model_format}.tflite"
                if model_format == "float32":
                    payload = convert_float32(saved_model_dir)
                elif model_format == "dynamic":
                    payload = convert_dynamic(saved_model_dir)
                elif model_format == "int8":
                    payload = convert_full_integer(
                        saved_model_dir, representative_samples, "int8"
                    )
                else:
                    payload = convert_full_integer(
                        saved_model_dir, representative_samples, "uint8"
                    )
                with tempfile.NamedTemporaryFile(
                    prefix=f".{model_path.name}.",
                    suffix=".pending",
                    dir=output_dir,
                    delete=False,
                ) as handle:
                    pending_path = Path(handle.name)
                    handle.write(payload)
                try:
                    if pending_path.stat().st_size < 16:
                        raise ExportError(
                            f"Converted model is unexpectedly small: {model_path.name}"
                        )
                    model_report = inspect_tflite(pending_path)
                    validate_tflite_format_report(model_report, model_format)
                    os.replace(pending_path, model_path)
                finally:
                    pending_path.unlink(missing_ok=True)

                model_report["path"] = model_path.name
                model_report["sha256"] = sha256_file(model_path)
                model_report["size_bytes"] = model_path.stat().st_size
                model_report["conversion_seconds"] = float(time.perf_counter() - started)
                model_report["reused_existing_file"] = False
                if source_model_sha256 is not None:
                    model_report["source_model_sha256"] = source_model_sha256
                if model_format in {"int8", "uint8"}:
                    model_report["representative_sha256"] = representative_sha256
                    model_report["comparison"] = compare_keras_and_tflite(
                        model,
                        model_path,
                        representative_samples[:comparison_samples],
                    )
                report["models"][model_format] = model_report
                artifacts[model_format] = model_path.name
                _emit(
                    progress,
                    stage_end,
                    f"{labels[model_format]} complete "
                    f"({model_report['conversion_seconds']:.1f} s).",
                )
    else:
        _emit(progress, 0.92, "Selected TensorFlow Lite files already exist; reusing them…")

    report["available_formats"] = sorted(report.get("models", {}).keys())
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifacts["conversion_report"] = report_path.name
    _emit(progress, 1.0, "TensorFlow Lite conversion complete.")
    return artifacts, report


def convert_all(
    model: Any,
    representative_samples: Sequence[np.ndarray],
    output_dir: Path,
    prefix: str,
    *,
    progress: ProgressFn | None = None,
    comparison_samples: int = 20,
) -> tuple[dict[str, str], dict[str, Any]]:
    return convert_selected(
        model,
        representative_samples,
        output_dir,
        prefix,
        VALID_TFLITE_FORMATS,
        progress=progress,
        comparison_samples=comparison_samples,
    )


def write_c_array(model_path: Path, output: Path, symbol: str) -> Path:
    payload = model_path.read_bytes()
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "",
        "#if defined(__cplusplus)",
        "  #define TM_LOCAL_ALIGNAS(x) alignas(x)",
        "#elif defined(_MSC_VER)",
        "  #define TM_LOCAL_ALIGNAS(x) __declspec(align(x))",
        "#else",
        "  #define TM_LOCAL_ALIGNAS(x) __attribute__((aligned(x)))",
        "#endif",
        "",
        f"TM_LOCAL_ALIGNAS(16) const unsigned char {symbol}[] = {{",
    ]
    for start in range(0, len(payload), 12):
        chunk = payload[start : start + 12]
        lines.append("    " + ", ".join(f"0x{value:02X}" for value in chunk) + ",")
    lines.extend(
        [
            "};",
            f"const unsigned int {symbol}_len = {len(payload)}u;",
            "#undef TM_LOCAL_ALIGNAS",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _input_note(kind: str) -> str:
    """One line describing what the exported model eats and emits."""

    notes = {
        "image": (
            "RGB image tensor; read conversion_report.json for dtype/scale/zero_point."
        ),
        "audio": (
            "Log-mel spectrogram tensor. Apply audio_frontend.json before inference."
        ),
        "abnormal_sound": (
            "96x64 YAMNet magnitude-log-mel patch. The model output is a "
            "1024-D embedding; apply yamnet_scorer.json and the RMS branch."
        ),
        "known_sound": (
            "96x64 YAMNet magnitude-log-mel patch. The model output is one INDEPENDENT "
            "score per class, in labels.txt order. The scores do not add up to 1 and are "
            "not calibrated probabilities."
        ),
    }
    if kind not in notes:
        raise ExportError(f"Unsupported project kind: {kind!r}.")
    return notes[kind]


def build_model_download(
    *,
    project: dict[str, Any],
    project_dir: Path,
    selected: Iterable[str],
    include_c_header: bool,
    output_zip: Path,
) -> Path:
    selected_set = {str(item).strip().lower() for item in selected}
    allowed = {*VALID_TFLITE_FORMATS, "keras"}
    unknown = sorted(selected_set.difference(allowed))
    if unknown:
        raise ExportError("Unsupported export format(s): " + ", ".join(unknown))
    if not selected_set:
        raise ExportError("Select at least one model format.")
    if project.get("training", {}).get("state") != "trained":
        raise ExportError("Train the model before exporting it.")

    artifacts = project.get("training", {}).get("artifacts", {})
    models_dir = Path(project_dir) / "models"
    missing: list[str] = []
    for key in selected_set:
        filename = artifacts.get(key)
        if not filename or not (models_dir / str(filename)).is_file():
            missing.append(key)
    if missing:
        raise ExportError(
            "Requested model format(s) were not generated: " + ", ".join(sorted(missing))
        )

    labels = [item["name"] for item in project["classes"]]
    kind = project["kind"]
    if kind == "image":
        prefix = "image_classifier"
    elif kind == "audio":
        prefix = "audio_classifier_spectrogram"
    elif kind == "abnormal_sound":
        prefix = "abnormal_sound_yamnet_embedding"
    elif kind == "known_sound":
        prefix = "known_sound_yamnet_classifier"
    else:
        raise ExportError(f"Unsupported project kind: {kind!r}.")

    with tempfile.TemporaryDirectory(prefix="tm_local_export_") as temp:
        root = Path(temp) / "export"
        root.mkdir(parents=True)
        for key in sorted(selected_set):
            filename = str(artifacts[key])
            shutil.copy2(models_dir / filename, root / filename)
        support_names = [
            "conversion_report.json",
            "training_report.json",
            "audio_frontend.json",
            "audio_frontend_reference.py",
        ]
        if kind == "abnormal_sound":
            support_names.extend(
                [
                    "yamnet_scorer.json",
                    "yamnet_frontend.json",
                    "yamnet_frontend_reference.py",
                    "contamination_report.json",
                ]
            )
        if kind == "known_sound":
            support_names.extend(
                ["yamnet_frontend.json", "yamnet_frontend_reference.py"]
            )
        if selected_set.intersection(VALID_TFLITE_FORMATS):
            support_names.extend(["run_model.py", "conversion_worker.log"])
        if kind == "known_sound":
            required_support = {"yamnet_frontend.json", "training_report.json"}
            if selected_set.intersection(VALID_TFLITE_FORMATS):
                required_support.update(
                    {
                        "conversion_report.json",
                        "yamnet_frontend_reference.py",
                        "run_model.py",
                    }
                )
            missing_support = sorted(
                name for name in required_support if not (models_dir / name).is_file()
            )
            if missing_support:
                raise ExportError(
                    "Known Sound export is missing required support artifact(s): "
                    + ", ".join(missing_support)
                    + ". Retrain or re-export the project."
                )
        if kind == "abnormal_sound":
            required_support = {
                "yamnet_scorer.json",
                "yamnet_frontend.json",
                "training_report.json",
            }
            if selected_set.intersection(VALID_TFLITE_FORMATS):
                required_support.update(
                    {
                        "conversion_report.json",
                        "yamnet_frontend_reference.py",
                        "run_model.py",
                    }
                )
            missing_support = sorted(
                name for name in required_support if not (models_dir / name).is_file()
            )
            if missing_support:
                raise ExportError(
                    "Abnormal Sound export is missing required support artifact(s): "
                    + ", ".join(missing_support)
                    + ". Retrain or re-export the project."
                )
        for support_name in support_names:
            source = models_dir / support_name
            if source.is_file():
                shutil.copy2(source, root / support_name)
        if kind != "abnormal_sound":
            (root / "labels.txt").write_text(
                "".join(f"{index} {label}\n" for index, label in enumerate(labels)),
                encoding="utf-8",
            )
        metadata = {
            "tool": "Teachable Machine Local Studio",
            "project_id": project["id"],
            "project_name": project["name"],
            "project_kind": kind,
            "selected_formats": sorted(selected_set),
            "generated_at": utc_now_iso(),
            "input_note": _input_note(kind),
        }
        if kind == "abnormal_sound":
            metadata.update(
                {
                    "detector_kind": "open_set_anomaly",
                    "detector_backend": "yamnet_embedding",
                    "model_output": "1024-D frozen YAMNet embedding (not class probabilities)",
                    "decision_artifact": "yamnet_scorer.json",
                    "evaluation_group_names": labels[1:],
                    "statement": (
                        "Reports deviation from the collected normal baseline; it does not "
                        "classify or name the anomaly."
                    ),
                }
            )
        else:
            metadata["labels"] = labels
        if kind == "audio":
            # Same argument as known_sound below, for the other kind that has a decision
            # policy: without it the ZIP says what the scores are but not when the student
            # decided a sound counts as heard, so a classmate re-running the model on
            # another host invents their own threshold and smoothing. kws_decision_policy()
            # is the one resolver the generated run_model.py also goes through. Imported
            # function-locally to keep this module's import cost off the TF-free paths.
            from .audio_pipeline import kws_decision_policy

            audio_settings = (
                (project.get("training") or {}).get("report") or {}
            ).get("settings") or {}
            metadata.update(kws_decision_policy(audio_settings))
        if kind == "known_sound":
            # The decision policy travels with the model: without it the ZIP says what the
            # scores are but not when the student decided a sound counts as heard. The
            # import is function-local because known_sound_pipeline imports ExportError
            # from this module.
            from .known_sound_pipeline import thresholds_for_labels

            report_settings = (
                (project.get("training") or {}).get("report") or {}
            ).get("settings") or {}
            metadata["detection_threshold"] = float(
                report_settings.get("detection_threshold", 0.5)
            )
            metadata["class_thresholds"] = thresholds_for_labels(report_settings, len(labels))
        (root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if include_c_header:
            int_model = root / f"{prefix}_int8.tflite"
            if int_model.is_file():
                write_c_array(int_model, root / "model_data.h", "g_model_data")
        # Written by export_service.ensure_export_artifacts() via merge_training_export();
        # empty for abnormal_sound/known_sound (no plain top-1 comparison) and whenever the
        # INT8/Float classifier agreement was above the acceptance threshold.
        conversion_warnings = (
            ((project.get("training") or {}).get("report") or {}).get("conversion") or {}
        ).get("warnings") or []
        readme = [
            "# Teachable Machine Local Studio model export",
            "",
            f"Project: {project['name']}",
            f"Type: {kind}",
            "",
            "## Files",
            "",
            "- `*_int8.tflite`: signed INT8 input/output model for MCU/NPU deployment.",
            "- `*_uint8.tflite`: UINT8 input/output compatibility model.",
            "- `*_float32.tflite`: Float32 reference model.",
            "- `*_dynamic.tflite`: dynamic-range quantized model.",
            "- `conversion_report.json`: tensor dtype, shape, quantization and operator audit.",
        ]
        if kind != "abnormal_sound":
            readme.append("- `labels.txt`: output class order.")
        if kind == "audio":
            readme.extend(
                [
                    "- `audio_frontend.json`: the exact PCM-to-log-mel settings used for training.",
                    "- `audio_frontend_reference.py`: reference feature extraction implementation.",
                    "",
                    "The INT8 audio model accepts a spectrogram, not raw PCM. The frontend must match exactly.",
                ]
            )
        elif kind == "abnormal_sound":
            readme.extend(
                [
                    "- `yamnet_frontend.json`: fixed 16 kHz / 64-mel / 96x64 frontend contract.",
                    "- `yamnet_frontend_reference.py`: exact PC reference frontend.",
                    "- `yamnet_scorer.json`: per-runtime normal reference, thresholds, RMS calibration, and binding hashes.",
                    "",
                    "The TFLite model outputs a 1024-D embedding, not Normal/Abnormal class probabilities.",
                    "Apply the matching runtime entry in yamnet_scorer.json and combine its embedding ratio with the RMS ratio.",
                    "Anomaly evaluation group names are never model output labels and never train the threshold.",
                    "Strict-integer conversion does not by itself prove that a particular MCU has enough flash, SRAM, or latency budget.",
                ]
            )
        if conversion_warnings:
            readme.extend(["", "## 注意", ""])
            readme.extend(f"- {message}" for message in conversion_warnings)
        (root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(root).as_posix())
    return output_zip
