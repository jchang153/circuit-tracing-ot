#!/usr/bin/env python
"""Print circuit-tracer feature visualization examples for selected CLT features."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_SCAN = "mntss/clt-gemma-2-2b-426k"


def cantor_pair(layer: int, feature_idx: int) -> int:
    total = int(layer) + int(feature_idx)
    return total * (total + 1) // 2 + int(feature_idx)


def cantor_unpair(feature_index: int) -> tuple[int, int]:
    z = int(feature_index)
    w = int(((8 * z + 1) ** 0.5 - 1) // 2)
    t = w * (w + 1) // 2
    y = z - t
    x = w - y
    return int(x), int(y)


def hf_url(scan: str, path: str) -> str:
    repo_id, sep, rest = str(scan).partition("//")
    if sep:
        file_path, _at, revision = rest.partition("@")
        prefix = f"{file_path}/" if file_path else ""
    else:
        repo_id, _at, revision = str(scan).partition("@")
        prefix = ""
    return (
        f"https://huggingface.co/{repo_id}/resolve/{revision or 'main'}/"
        f"{prefix}features/{path}"
    )


def fetch_bytes(url: str, *, range_header: str | None = None) -> bytes:
    headers = {"User-Agent": "circuit-tracing-ot-feature-examples/1.0"}
    if range_header:
        headers["Range"] = range_header
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


@lru_cache(maxsize=8)
def load_index(scan: str) -> Any:
    raw = fetch_bytes(hf_url(scan, "index.json.gz"))
    return json.loads(gzip.decompress(raw).decode("utf-8"))


def layer_index_entry(index: Any, layer: int) -> dict[str, Any]:
    if isinstance(index, list):
        return index[int(layer)]
    if isinstance(index, dict):
        return index[str(int(layer))]
    raise TypeError(f"Unsupported feature index type: {type(index).__name__}")


def load_feature(scan: str, feature_index: int) -> dict[str, Any]:
    layer, feature_idx = cantor_unpair(feature_index)
    index = load_index(scan)
    layer_entry = layer_index_entry(index, int(layer))
    offsets = layer_entry["offsets"]
    filename = layer_entry["filename"]
    start = int(offsets[int(feature_idx)])
    end = int(offsets[int(feature_idx) + 1])
    raw = fetch_bytes(hf_url(scan, filename), range_header=f"bytes={start}-{end}")
    data_length = raw[0] | (raw[1] << 8) | (raw[2] << 16) | (raw[3] << 24)
    compressed = raw[4 : 4 + data_length]
    return json.loads(zlib.decompress(compressed, wbits=47).decode("utf-8"))


def token_text(tokens: list[str], values: list[float], *, min_abs_activation: float) -> str:
    pieces = []
    for token, value in zip(tokens, values):
        clean = str(token).replace("\n", "\\n")
        if abs(float(value)) >= float(min_abs_activation):
            pieces.append(f"[{clean}:{float(value):.3g}]")
        else:
            pieces.append(clean)
    return "".join(pieces)


def sorted_quantiles(feature: dict[str, Any]) -> list[dict[str, Any]]:
    quantiles = list(feature.get("examples_quantiles", []))
    return sorted(quantiles, key=lambda item: str(item.get("quantile_name", "")), reverse=True)


def print_feature(
    *,
    scan: str,
    layer: int,
    feature_idx: int,
    top_examples: int,
    quantiles: int,
    min_abs_activation: float,
) -> None:
    feature_index = cantor_pair(layer, feature_idx)
    feature = load_feature(scan, feature_index)
    print(f"\n=== {scan} | L{layer} f{feature_idx} | global {feature_index} ===")
    print(
        "activation_frequency="
        f"{feature.get('activation_frequency', 'n/a')} "
        f"act_min={feature.get('act_min', 'n/a')} "
        f"act_max={feature.get('act_max', 'n/a')}"
    )
    top_logits = feature.get("top_logits", [])
    bottom_logits = feature.get("bottom_logits", [])
    if top_logits:
        print("top_logits:", ", ".join(str(item) for item in top_logits[:10]))
    if bottom_logits:
        print("bottom_logits:", ", ".join(str(item) for item in bottom_logits[:10]))
    shown_quantiles = 0
    for quantile in sorted_quantiles(feature):
        examples = list(quantile.get("examples", []))
        if not examples:
            continue
        shown_quantiles += 1
        print(f"\n-- {quantile.get('quantile_name', 'examples')} --")
        for index, example in enumerate(examples[: int(top_examples)], start=1):
            tokens = [str(token) for token in example.get("tokens", [])]
            values = [float(value) for value in example.get("tokens_acts_list", [])]
            max_activation = max(values, key=lambda value: abs(value)) if values else 0.0
            print(f"{index}. max_abs_activation={abs(float(max_activation)):.4g}")
            print(token_text(tokens, values, min_abs_activation=min_abs_activation))
        if shown_quantiles >= int(quantiles):
            break


def parse_feature_arg(value: str) -> tuple[int, int]:
    text = str(value).strip().lower().replace("layer", "l").replace("feature", "f")
    text = text.replace(" ", "")
    if text.startswith("l") and "f" in text:
        layer_text, feature_text = text[1:].split("f", maxsplit=1)
        return int(layer_text), int(feature_text)
    if ":" in text:
        layer_text, feature_text = text.split(":", maxsplit=1)
        return int(layer_text), int(feature_text)
    raise ValueError(f"Feature must look like L18f15532 or 18:15532, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", default=DEFAULT_SCAN)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--layer", type=int)
    parser.add_argument("--feature-idx", type=int)
    parser.add_argument("--feature-index", type=int, help="Global Cantor feature id.")
    parser.add_argument("--top-examples", type=int, default=4)
    parser.add_argument("--quantiles", type=int, default=2)
    parser.add_argument("--min-abs-activation", type=float, default=0.01)
    parser.add_argument("--from-probe-results", type=Path)
    parser.add_argument("--target", choices=("answer_pointer", "answer_token"))
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def selected_features_from_args(args: argparse.Namespace) -> list[tuple[int, int]]:
    features = [parse_feature_arg(value) for value in args.feature]
    if args.layer is not None or args.feature_idx is not None:
        if args.layer is None or args.feature_idx is None:
            raise ValueError("--layer and --feature-idx must be provided together.")
        features.append((int(args.layer), int(args.feature_idx)))
    if args.feature_index is not None:
        features.append(cantor_unpair(int(args.feature_index)))
    if args.from_probe_results:
        if not args.target:
            raise ValueError("--target is required with --from-probe-results.")
        payload = json.loads(
            (args.from_probe_results / f"{args.target}_results.json").read_text(encoding="utf-8")
        )
        layer = int(payload["selected_layer"])
        for item in payload["feature_ranking"][: int(args.top_n)]:
            features.append((layer, int(item["feature_idx"])))
    return features


def main() -> None:
    args = parse_args()
    features = selected_features_from_args(args)
    if not features:
        print("Provide --feature L18f15532, --layer/--feature-idx, or --from-probe-results.", file=sys.stderr)
        raise SystemExit(2)
    for layer, feature_idx in features:
        print_feature(
            scan=str(args.scan),
            layer=int(layer),
            feature_idx=int(feature_idx),
            top_examples=int(args.top_examples),
            quantiles=int(args.quantiles),
            min_abs_activation=float(args.min_abs_activation),
        )


if __name__ == "__main__":
    main()
