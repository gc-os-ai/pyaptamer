#!/usr/bin/env python3
"""Convert deepDNAshape TensorFlow checkpoints to PyTorch .pt files.

Expects a local folder of TF checkpoints, one feature per prefix::

    MGW.index
    MGW.data-00000-of-00001
    ProT.index
    ...

Writes ``Feature.pt`` state dicts for ``pyaptamer.deepdnashape._model.DNAModel``.

Usage
-----
::

    # Interactive (prompts for input / output folders)
    python build_tools/convert_deepdnashape.py

    # Or pass paths on the CLI
    python build_tools/convert_deepdnashape.py \\
        --models-dir /path/to/tf_weights \\
        --outdir converted_weights
"""

from __future__ import annotations

__author__ = ["Alleny244"]

import argparse
import sys
from pathlib import Path

import numpy as np

INTRABASE_FEATURES = {
    "Shear",
    "Stretch",
    "Stagger",
    "Buckle",
    "ProT",
    "Opening",
    "MGW",
    "EP",
    "Shear-FL",
    "Stretch-FL",
    "Stagger-FL",
    "Buckle-FL",
    "ProT-FL",
    "Opening-FL",
    "MGW-FL",
}

INTERBASE_FEATURES = {
    "Shift",
    "Slide",
    "Rise",
    "Tilt",
    "Roll",
    "HelT",
    "Shift-FL",
    "Slide-FL",
    "Rise-FL",
    "Tilt-FL",
    "Roll-FL",
    "HelT-FL",
}

ALL_FEATURES = sorted(INTRABASE_FEATURES | INTERBASE_FEATURES)
MP_LAYERS = 7


def _mp_var_mapping(i: int) -> list[tuple[str, str]]:
    """TF variable name → PyTorch state-dict key for message-passing layer i."""
    tf = f"mp_layers/{i}"
    pt = f"mp.{i}"
    return [
        (f"{tf}/Weight1", f"{pt}.w_next"),
        (f"{tf}/Weight2", f"{pt}.w_prev"),
        (f"{tf}/Bias1", f"{pt}.b"),
        (f"{tf}/Weight1_term2", f"{pt}.w_next_all"),
        (f"{tf}/Weight2_term2", f"{pt}.w_prev_all"),
        (f"{tf}/BiasAll", f"{pt}.b_all"),
        (f"{tf}/bn/gamma", f"{pt}.bn.weight"),
        (f"{tf}/bn/beta", f"{pt}.bn.bias"),
        (f"{tf}/bn/moving_mean", f"{pt}.bn.running_mean"),
        (f"{tf}/bn/moving_variance", f"{pt}.bn.running_var"),
        (f"{tf}/gru_layer/kernel", f"{pt}.gru.kernel"),
        (f"{tf}/gru_layer/recurrent_kernel", f"{pt}.gru.rec_kernel"),
        (f"{tf}/gru_layer/bias", f"{pt}.gru.bias"),
    ]


def _prompt_path(prompt: str, *, must_exist: bool = True) -> Path:
    while True:
        raw = input(prompt).strip().strip("'\"")
        if not raw:
            print("  Path cannot be empty.")
            continue
        path = Path(raw).expanduser().resolve()
        if must_exist and not path.is_dir():
            print(f"  Not a directory: {path}")
            continue
        return path


def _resolve_dir(value: str | None, prompt: str, *, must_exist: bool = True) -> Path:
    if value is None:
        return _prompt_path(prompt, must_exist=must_exist)
    path = Path(value).expanduser().resolve()
    if must_exist and not path.is_dir():
        raise SystemExit(f"Not a directory: {path}")
    return path


def _discover_features(models_dir: Path, requested: list[str] | None) -> list[str]:
    """Return feature names that have a ``.index`` file in models_dir."""
    if requested:
        return requested
    found = sorted(p.stem for p in models_dir.glob("*.index"))
    if not found:
        raise SystemExit(f"No *.index TF checkpoints found in {models_dir}")
    unknown = [f for f in found if f not in ALL_FEATURES]
    if unknown:
        print(f"Note: unknown feature names (will still try): {unknown}")
    return found


def _read_tf_checkpoint(ckpt_path: str) -> dict[str, np.ndarray]:
    """Load one TF checkpoint → {tf_var_name: numpy array}."""
    import tensorflow as tf

    reader = tf.train.load_checkpoint(ckpt_path)
    var_map = reader.get_variable_to_shape_map()
    suffix = (
        "/.ATTRIBUTES/VARIABLE_VALUE"
        if any(".ATTRIBUTES" in k for k in var_map)
        else ""
    )

    def get(name: str) -> np.ndarray:
        for candidate in (f"{name}{suffix}", name):
            if candidate in var_map:
                return reader.get_tensor(candidate).copy()
        raise KeyError(f"Variable not found: {name}")

    arrays: dict[str, np.ndarray] = {}
    for conv_prefix in ("selfconv", "conv1d", "conv1_d"):
        try:
            arrays["selfconv/kernel"] = get(f"{conv_prefix}/kernel")
            arrays["selfconv/bias"] = get(f"{conv_prefix}/bias")
            break
        except KeyError:
            continue
    else:
        raise KeyError(f"Cannot find conv kernel in {ckpt_path}")

    for i in range(MP_LAYERS):
        for tf_name, _ in _mp_var_mapping(i):
            arrays[tf_name] = get(tf_name)
    return arrays


def _arrays_to_state_dict(arrays: dict[str, np.ndarray]):
    """Map raw TF arrays → PyTorch state dict (with Conv1D axis fix)."""
    import torch

    state_dict = {
        # TF Conv1D: (K, Cin, Cout) → PT: (Cout, Cin, K)
        "input_conv.weight": torch.from_numpy(
            arrays["selfconv/kernel"].transpose(2, 1, 0).copy()
        ),
        "input_conv.bias": torch.from_numpy(arrays["selfconv/bias"].copy()),
    }
    for i in range(MP_LAYERS):
        for tf_name, pt_name in _mp_var_mapping(i):
            state_dict[pt_name] = torch.from_numpy(arrays[tf_name].copy())
        state_dict[f"mp.{i}.bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    return state_dict


def _validate(state_dict, feature: str) -> None:
    """Load weights into DNAModel to confirm key/shape match."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pyaptamer.deepdnashape._model import DNAModel

    input_features = 4 if feature in INTRABASE_FEATURES else 16
    model = DNAModel(input_features=input_features)
    model.load_state_dict(state_dict)
    model.eval()
    seq_len = 10 if feature in INTRABASE_FEATURES else 9
    model(torch.randn(seq_len, input_features))


def convert(
    models_dir: Path,
    outdir: Path,
    features: list[str] | None = None,
    *,
    validate: bool = False,
) -> int:
    """Convert all TF checkpoints in models_dir → .pt files in outdir."""
    import torch

    features = _discover_features(models_dir, features)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {models_dir}")
    print(f"Output: {outdir}")
    print(f"Features ({len(features)}): {', '.join(features)}\n")

    succeeded, failed = [], []
    for feature in features:
        index_file = models_dir / f"{feature}.index"
        if not index_file.exists():
            print(f"  SKIP  {feature}: missing .index")
            failed.append((feature, "missing checkpoint"))
            continue

        print(f"  {feature} ...", end=" ", flush=True)
        try:
            arrays = _read_tf_checkpoint(str(models_dir / feature))
            state_dict = _arrays_to_state_dict(arrays)
            if validate:
                _validate(state_dict, feature)
            out_path = outdir / f"{feature}.pt"
            torch.save(state_dict, out_path)
            print(f"OK → {out_path.name}")
            succeeded.append(feature)
        except Exception as exc:
            print(f"FAILED ({exc})")
            failed.append((feature, str(exc)))

    print(f"\nDone: {len(succeeded)} ok, {len(failed)} failed.")
    if failed:
        for feat, reason in failed:
            print(f"  - {feat}: {reason}")
    return 0 if succeeded and not failed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a folder of deepDNAshape TF checkpoints into PyTorch .pt files."
        )
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Folder with Feature.index / Feature.data-* files. Prompted if omitted.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Folder to write .pt files into. Prompted if omitted.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional subset of feature names. Default: all *.index in models-dir.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Load each .pt into DNAModel after conversion (needs pyaptamer + torch).",
    )
    args = parser.parse_args()

    models_dir = _resolve_dir(
        args.models_dir,
        "Path to folder with TF checkpoints (.index + .data): ",
    )
    outdir = _resolve_dir(
        args.outdir,
        "Path to output folder for .pt files: ",
        must_exist=False,
    )
    raise SystemExit(convert(models_dir, outdir, args.features, validate=args.validate))


if __name__ == "__main__":
    main()
