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
graphs/
graph_files/
runs/
```

These are ignored by git because raw graphs and exported graph files can be large.

## Serve Graphs

```bash
python scripts/serve_graphs.py --graph-file-dir graph_files --port 8046
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
