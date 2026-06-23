#!/usr/bin/env python
"""Export an HTML dashboard for probe-selected features using viewer CLERP labels."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


NODE_ID_RE = re.compile(r"^(?P<layer>\d+)_(?P<feature>\d+)_(?P<ctx>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-results-dir", type=Path, required=True)
    parser.add_argument(
        "--graph-json",
        type=Path,
        action="append",
        required=True,
        help="Exported circuit-tracer viewer JSON. Can be passed multiple times.",
    )
    parser.add_argument("--targets", default="answer_pointer,answer_token")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--slug", default="probe_feature_visualizations")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_clerps(graph: dict[str, Any]) -> dict[str, str]:
    raw = graph.get("qParams", {}).get("clerps", "[]")
    if not raw:
        return {}
    try:
        pairs = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {}
    labels: dict[str, str] = {}
    for pair in pairs:
        if isinstance(pair, list) and len(pair) >= 2:
            labels[str(pair[0])] = str(pair[1])
    return labels


def parse_supernodes(graph: dict[str, Any]) -> dict[str, list[str]]:
    raw = graph.get("qParams", {}).get("supernodes", "[]")
    if not raw:
        return {}
    try:
        groups = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {}
    labels_by_node: dict[str, list[str]] = {}
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            continue
        label = str(group[0])
        for node_id in group[1:]:
            labels_by_node.setdefault(str(node_id), []).append(label)
    return labels_by_node


def graph_slug(path: Path, graph: dict[str, Any]) -> str:
    metadata = graph.get("metadata", {})
    return str(metadata.get("slug") or path.stem)


def local_feature_from_node(node: dict[str, Any]) -> tuple[int, int, int] | None:
    match = NODE_ID_RE.match(str(node.get("node_id", "")))
    if not match:
        return None
    return (
        int(match.group("layer")),
        int(match.group("feature")),
        int(match.group("ctx")),
    )


def index_graph(path: Path) -> dict[str, Any]:
    graph = load_json(path)
    metadata = graph.get("metadata", {})
    tokens = metadata.get("prompt_tokens", [])
    if not isinstance(tokens, list):
        tokens = []
    clerp_by_global = parse_clerps(graph)
    supernodes = parse_supernodes(graph)
    pinned_ids = {
        item.strip()
        for item in str(graph.get("qParams", {}).get("pinnedIds", "")).split(",")
        if item.strip()
    }
    indexed: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict) or node.get("feature_type") != "cross layer transcoder":
            continue
        parsed = local_feature_from_node(node)
        if parsed is None:
            continue
        layer, local_feature, ctx_idx = parsed
        global_feature = str(node.get("feature", ""))
        token = tokens[ctx_idx] if 0 <= ctx_idx < len(tokens) else ""
        node_id = str(node.get("node_id", ""))
        label = str(node.get("clerp") or clerp_by_global.get(global_feature, ""))
        indexed.setdefault((layer, local_feature), []).append(
            {
                "graph_path": str(path),
                "graph_slug": graph_slug(path, graph),
                "node_id": node_id,
                "layer": layer,
                "feature_idx": local_feature,
                "ctx_idx": ctx_idx,
                "token": str(token),
                "global_feature": global_feature,
                "clerp": label,
                "activation": float(node.get("activation", 0.0) or 0.0),
                "influence": float(node.get("influence", 0.0) or 0.0),
                "supernodes": supernodes.get(node_id, []),
                "pinned": node_id in pinned_ids,
                "prompt": str(metadata.get("prompt", "")),
            }
        )
    for matches in indexed.values():
        matches.sort(key=lambda item: (-abs(float(item["activation"])), int(item["ctx_idx"])))
    return {
        "path": str(path),
        "slug": graph_slug(path, graph),
        "prompt": str(metadata.get("prompt", "")),
        "prompt_tokens": tokens,
        "features": indexed,
        "clerp_count": len(clerp_by_global),
        "node_count": len(graph.get("nodes", [])),
    }


def load_probe_features(
    *,
    probe_results_dir: Path,
    targets: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target in targets:
        path = probe_results_dir / f"{target}_results.json"
        payload = load_json(path)
        selected_layer = int(payload["selected_layer"])
        selected_probe = payload.get("selected_probe", {})
        for item in payload.get("feature_ranking", [])[: int(top_n)]:
            records.append(
                {
                    "target": target,
                    "selected_layer": selected_layer,
                    "feature_idx": int(item["feature_idx"]),
                    "rank": int(item["rank"]),
                    "column": int(item["column"]),
                    "selection_frequency": float(item.get("selection_frequency", 0.0)),
                    "main_centered_column_l2": float(item.get("main_centered_column_l2", 0.0)),
                    "mean_centered_column_l2": float(item.get("mean_centered_column_l2", 0.0)),
                    "selected_probe": selected_probe,
                }
            )
    return records


def build_dashboard_data(
    *,
    probe_results_dir: Path,
    graph_jsons: list[Path],
    targets: list[str],
    top_n: int,
) -> dict[str, Any]:
    probe_features = load_probe_features(
        probe_results_dir=probe_results_dir,
        targets=targets,
        top_n=top_n,
    )
    graphs = [index_graph(path) for path in graph_jsons]
    rows: list[dict[str, Any]] = []
    for feature in probe_features:
        key = (int(feature["selected_layer"]), int(feature["feature_idx"]))
        matches: list[dict[str, Any]] = []
        for graph in graphs:
            matches.extend(graph["features"].get(key, []))
        matches.sort(
            key=lambda item: (
                not bool(item["clerp"]),
                -abs(float(item["activation"])),
                str(item["graph_slug"]),
                int(item["ctx_idx"]),
            )
        )
        best_label = next((item["clerp"] for item in matches if item["clerp"]), "")
        rows.append({**feature, "clerp": best_label, "matches": matches})
    return {
        "probe_results_dir": str(probe_results_dir),
        "graphs": [
            {
                "path": graph["path"],
                "slug": graph["slug"],
                "node_count": graph["node_count"],
                "clerp_count": graph["clerp_count"],
                "prompt": graph["prompt"],
            }
            for graph in graphs
        ],
        "targets": targets,
        "top_n": int(top_n),
        "features": rows,
    }


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float) -> str:
    return f"{float(value):.4g}"


def render_html(data: dict[str, Any]) -> str:
    total = len(data["features"])
    matched = sum(1 for row in data["features"] if row["matches"])
    labeled = sum(1 for row in data["features"] if row["clerp"])
    rows_html = []
    for row in data["features"]:
        matches = row["matches"]
        top = matches[0] if matches else {}
        empty_details = (
            '<tr><td colspan="7">'
            "Feature was not present in the supplied viewer graph JSON files."
            "</td></tr>"
        )
        details = "".join(
            "<tr>"
            f"<td>{esc(match['graph_slug'])}</td>"
            f"<td>{esc(match['node_id'])}</td>"
            f"<td>{esc(match['token'])}</td>"
            f"<td>{fmt(match['activation'])}</td>"
            f"<td>{fmt(match['influence'])}</td>"
            f"<td>{esc(match['clerp'])}</td>"
            f"<td>{esc(', '.join(match['supernodes']))}</td>"
            "</tr>"
            for match in matches[:20]
        )
        rows_html.append(
            "<details class='feature-card' "
            f"data-target='{esc(row['target'])}' "
            f"data-text='{esc(str(row['target']) + ' ' + str(row['feature_idx']) + ' ' + row['clerp'])}'>"
            "<summary>"
            "<span class='rank'>"
            f"{esc(row['target'])} #{int(row['rank'])}"
            "</span>"
            "<span class='feature'>"
            f"L{int(row['selected_layer'])} f{int(row['feature_idx'])}"
            "</span>"
            "<span class='label'>"
            f"{esc(row['clerp'] or 'no viewer label found')}"
            "</span>"
            "<span class='metric'>"
            f"{len(matches)} match{'es' if len(matches) != 1 else ''}"
            "</span>"
            "</summary>"
            "<div class='feature-body'>"
            "<div class='meta-grid'>"
            f"<div><b>Selection frequency</b><span>{fmt(row['selection_frequency'])}</span></div>"
            f"<div><b>Main centered column L2</b><span>{fmt(row['main_centered_column_l2'])}</span></div>"
            f"<div><b>Mean bootstrap L2</b><span>{fmt(row['mean_centered_column_l2'])}</span></div>"
            f"<div><b>Top token</b><span>{esc(top.get('token', ''))}</span></div>"
            "</div>"
            "<table><thead><tr>"
            "<th>Graph</th><th>Node</th><th>Token</th><th>Activation</th><th>Influence</th><th>Viewer label</th><th>Supernodes</th>"
            "</tr></thead><tbody>"
            f"{details or empty_details}"
            "</tbody></table>"
            "</div></details>"
        )
    graph_rows = "".join(
        "<tr>"
        f"<td>{esc(graph['slug'])}</td>"
        f"<td>{esc(graph['path'])}</td>"
        f"<td>{int(graph['node_count'])}</td>"
        f"<td>{int(graph['clerp_count'])}</td>"
        "</tr>"
        for graph in data["graphs"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Probe Feature Visualizations</title>
  <style>
    :root {{
      --bg: #f6f7f8;
      --panel: #ffffff;
      --text: #172026;
      --muted: #5d6972;
      --line: #d9dee3;
      --accent: #126a5f;
      --accent-2: #8b4a14;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 24px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 700; letter-spacing: 0; }}
    .subtle {{ color: var(--muted); max-width: 980px; }}
    main {{ padding: 20px 28px 36px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .card b {{ display: block; font-size: 22px; margin-bottom: 2px; }}
    .card span {{ color: var(--muted); }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; margin: 16px 0; flex-wrap: wrap; }}
    input, select {{
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: white;
      color: var(--text);
    }}
    input {{ min-width: min(420px, 100%); }}
    .feature-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 10px 0;
      overflow: hidden;
    }}
    summary {{
      cursor: pointer;
      display: grid;
      grid-template-columns: 140px 110px minmax(180px, 1fr) 90px;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
    }}
    .rank {{ color: var(--accent); font-weight: 700; }}
    .feature {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .label {{ font-weight: 600; overflow-wrap: anywhere; }}
    .metric {{ color: var(--muted); text-align: right; }}
    .feature-body {{ border-top: 1px solid var(--line); padding: 12px 14px 14px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }}
    .meta-grid div {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfcfc; }}
    .meta-grid b {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; background: #fbfcfc; }}
    td {{ overflow-wrap: anywhere; }}
    .sources {{ margin-top: 24px; }}
    @media (max-width: 820px) {{
      main, header {{ padding-left: 14px; padding-right: 14px; }}
      .cards, .meta-grid {{ grid-template-columns: 1fr 1fr; }}
      summary {{ grid-template-columns: 1fr; gap: 4px; }}
      .metric {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Probe Feature Visualizations</h1>
    <div class="subtle">Probe-selected AP/AT features joined to exported circuit-tracer viewer JSON. Labels come from the same CLERP feature visualization metadata used by the attribution graph viewer.</div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><b>{total}</b><span>probe features</span></div>
      <div class="card"><b>{matched}</b><span>present in supplied graphs</span></div>
      <div class="card"><b>{labeled}</b><span>with viewer labels</span></div>
      <div class="card"><b>{len(data['graphs'])}</b><span>viewer graph files</span></div>
    </section>
    <section class="toolbar">
      <input id="search" placeholder="Search target, feature id, or label">
      <select id="target">
        <option value="">All targets</option>
        <option value="answer_pointer">answer_pointer</option>
        <option value="answer_token">answer_token</option>
      </select>
    </section>
    <section id="features">
      {''.join(rows_html)}
    </section>
    <section class="sources">
      <h2>Sources</h2>
      <table><thead><tr><th>Graph</th><th>Path</th><th>Nodes</th><th>CLERP labels</th></tr></thead><tbody>{graph_rows}</tbody></table>
    </section>
  </main>
  <script>
    const search = document.getElementById('search');
    const target = document.getElementById('target');
    const cards = Array.from(document.querySelectorAll('.feature-card'));
    function applyFilters() {{
      const q = search.value.trim().toLowerCase();
      const t = target.value;
      for (const card of cards) {{
        const okTarget = !t || card.dataset.target === t;
        const okText = !q || card.dataset.text.toLowerCase().includes(q);
        card.style.display = okTarget && okText ? '' : 'none';
      }}
    }}
    search.addEventListener('input', applyFilters);
    target.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data = build_dashboard_data(
        probe_results_dir=args.probe_results_dir,
        graph_jsons=args.graph_json,
        targets=parse_csv(args.targets),
        top_n=int(args.top_n),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"{args.slug}.json"
    html_path = args.output_dir / f"{args.slug}.html"
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(render_html(data), encoding="utf-8")
    print(f"wrote {html_path}")
    print(f"wrote {data_path}")


if __name__ == "__main__":
    main()
