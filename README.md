# circuit-tracing-ot

Thin MCQA wrapper around `circuit-tracer` for Gemma-2-2B CLT attribution graphs.

This repo intentionally starts with the three workflows implemented by
`safety-research/circuit-tracer`:

1. Build attribution graphs from a model plus pretrained transcoders.
2. Export and serve pruned graph files for the interactive HTML viewer.
3. Run feature interventions on selected transcoder features.

The goal is to run the upstream `circuit-tracer` attribution graph and feature-intervention
pipeline on CopyColors MCQA prompts with as little new machinery as possible.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[viz]"
```

On NCSA Delta, install a CUDA wheel compatible with the cluster driver before running traces:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Gemma-2-2B requires Hugging Face access. Set one of:

```bash
export HF_TOKEN=...
export HUGGING_FACE_HUB_TOKEN=...
```

## Transcoder Sets

The default is the smaller Gemma-2-2B CLT:

```text
mntss/clt-gemma-2-2b-426k
```

Switch to the larger CLT with:

```bash
--transcoder-set mntss/clt-gemma-2-2b-2.5M
```

or use the shortcut:

```bash
--transcoder-size 2.5m
```

## Trace Representative MCQA Prompts

Prompts are loaded directly from the Hugging Face dataset:

```text
jchang153/copycolors_mcqa
```

The default split is `train`. The live dataset rows already include a formatted `prompt` plus
`choices`, so the tracing code passes that prompt directly to `circuit-tracer`:

```text
Question: {object} is {color}. What color is {object}?
A. {choice text}
B. {choice text}
...
Answer:
```

Trace the first training prompt:

```bash
python scripts/trace_representative_mcqa.py
```

By default this keeps 1,000 selected feature nodes, exports the viewer graph with
`node_threshold=1.0` and `edge_threshold=1.0`, and writes a flat result directory:

```text
results/delta_default_train_0/
  default-train-0.pt
  default-train-0.json
  default-train-0.manifest.json
  graph-metadata.json
```

`graph-metadata.json` is enriched after export with raw and viewer graph statistics, including
selected feature-node counts, raw adjacency-matrix size, viewer node/link counts, counts by node
type, per-layer/per-position decompositions, and node IDs grouped by type.

Trace a specific dataset row by row id or formatted prompt id:

```bash
python scripts/trace_representative_mcqa.py --prompt-id 21
python scripts/trace_representative_mcqa.py --prompt-id default-train-21
```

Trace all loaded prompts, optionally limiting the number of rows:

```bash
python scripts/trace_representative_mcqa.py --all
python scripts/trace_representative_mcqa.py --all --limit 4
```

If you point the script at an older multi-config CopyColors dataset, pass the config name:

```bash
python scripts/trace_representative_mcqa.py \
  --dataset-config 10_answer_choices \
  --dataset-split validation
```

Outputs are written under:

```text
results/delta_default_train_0/
```

These outputs are ignored by git because raw graphs and exported graph files can be large.

To copy a completed Delta run back to your laptop from a local terminal:

```bash
rsync -avz jchang6@login.delta.ncsa.illinois.edu:~/circuit-tracing-ot/results/delta_default_train_0/ \
  /Users/jchang153/Documents/GitHub/circuit-tracing-ot/results/delta_default_train_0/
```

## Serve Graphs

```bash
python scripts/serve_graphs.py \
  --graph-file-dir results/delta_default_train_0 \
  --port 8046
```

Then open:

```text
http://localhost:8046/index.html
```

The graph UI supports the same annotation workflow as `circuit-tracer`: select nodes, pin nodes,
edit annotations, and group nodes into supernodes.

## Interventions

After inspecting a graph, choose one or more feature IDs from the UI and run:

```bash
python scripts/intervene_mcqa.py \
  --prompt-id 21 \
  --feature 20:-1:341:0.0
```

Feature tuples are:

```text
layer:position:feature_idx:new_value
```

Use `position=-1` for the final prompt position, matching the demo notebooks.

## MCQA PLOT over CLT Features

The CLT PLOT runner ports the MCQA experiment protocol from
`jchang153/causal-abstractions-ot` and changes only the neural intervention backend:
counterfactuals, factual filtering, `D_train`/`D_cal`/`D_te` split construction, Sinkhorn OT/UOT,
calibration, and test evaluation follow the original MCQA PLOT code. Instead of residual-stream
swaps, candidate sites call `ReplacementModel.feature_intervention` on CLT feature values.

Run layer localization, then within-layer feature localization:

```bash
python scripts/plot_mcqa_clt.py \
  --dataset-path jchang153/copycolors_mcqa \
  --dataset-size 2000 \
  --split-seed 0 \
  --train-pool-size 200 \
  --calibration-pool-size 100 \
  --test-pool-size 100 \
  --layers 0-25 \
  --token-position-id last_token \
  --stage-a-transport-methods uot \
  --ot-epsilons 0.5,1.0,2.0,4.0 \
  --uot-beta-neurals 0.1,0.3,1.0,3.0 \
  --stage-a-row-top-k 6 \
  --top-layers 4
```

For a Stage A-only layer-ranking run, add:

```bash
  --skip-stage-b \
  --results-timestamp stage_a_layers
```

The result file is:

```text
results/mcqa_clt_mcqa_plot_clt/mcqa_plot_clt_results.json
```

With `--results-timestamp stage_a_layers`, the result file is:

```text
results/stage_a_layers_mcqa_plot_clt/mcqa_plot_clt_results.json
```

Stage A enumerates CLT layer sites at `last_token`, copies the full extracted source CLT feature
layer into the base prompt, builds train-set signatures, solves the same OT/UOT transport rows as
the original MCQA PLOT layer sweep, shortlists the top transport sites, evaluates those sites on
`D_cal`, selects the best calibrated layer site, and evaluates it on `D_te`. Stage B enumerates
top-activating CLT feature sites inside the selected layers, reuses
the same signature/transport/calibration/test logic, and reports the selected features.
For Stage A-only analysis, inspect `stage_a.layer_rankings_by_var`: it ranks the best calibrated
CLT layer sites separately for `answer_pointer` and `answer_token`.

To visualize the PLOT-selected features in the existing circuit-tracer viewer without applying
additional pruning, first produce an unpruned trace for a representative prompt:

```bash
python scripts/trace_representative_mcqa.py \
  --prompt-id 0 \
  --max-feature-nodes 5000 \
  --node-threshold 0.0 \
  --edge-threshold 0.0 \
  --result-dir results/unpruned_default_train_0
```

Then export a viewer graph centered on the selected PLOT features:

```bash
python scripts/export_plot_feature_graph.py \
  --plot-results results/mcqa_clt_mcqa_plot_clt/mcqa_plot_clt_results.json \
  --graph-json results/unpruned_default_train_0/default-train-0.json \
  --graph-metadata results/unpruned_default_train_0/graph-metadata.json \
  --output-dir results/plot_mcqa_clt/viewer \
  --include-non-feature-nodes
```

Serve it with the same viewer:

```bash
python scripts/serve_graphs.py --graph-file-dir results/plot_mcqa_clt/viewer --port 8046
```

## Notebook

Open:

```text
notebooks/01_mcqa_circuit_tracing.ipynb
```

It mirrors the upstream `attribute_demo.ipynb` and `intervention_demo.ipynb` flow:

1. load `ReplacementModel`
2. run `attribute`
3. save a raw graph
4. call `create_graph_files`
5. serve the HTML viewer
6. run a feature intervention

## Upstream References

- `safety-research/circuit-tracer`
- `demos/attribute_demo.ipynb`
- `demos/attribution_targets_demo.ipynb`
- `demos/intervention_demo.ipynb`
- `demos/circuit_tracing_tutorial.ipynb`
