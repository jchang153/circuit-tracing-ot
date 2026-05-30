# circuit-tracing-ot

Thin MCQA wrapper around `circuit-tracer` for Gemma-2-2B CLT attribution graphs.

This repo intentionally starts with the three workflows implemented by
`safety-research/circuit-tracer`:

1. Build attribution graphs from a model plus pretrained transcoders.
2. Export and serve pruned graph files for the interactive HTML viewer.
3. Run feature interventions on selected transcoder features.

OT, counterfactual alignment, progressive localization, and DAS are planned follow-ups. The first
goal is to get representative MCQA prompts into the existing CLT attribution-graph stack with as
little new machinery as possible.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[viz]"
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

```bash
python scripts/trace_representative_mcqa.py --prompt-id copycolors-a
```

Trace all configured prompts:

```bash
python scripts/trace_representative_mcqa.py --all
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
  --prompt-id copycolors-a \
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
