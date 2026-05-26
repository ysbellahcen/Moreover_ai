# Moreover.ai

**AI Infrastructure Workload Sizing Framework**

Moreover.ai is an open-source framework for estimating which AI workloads can realistically run on a given GPU infrastructure. It goes beyond simple VRAM checks to give infrastructure teams a transparent, reproducible, and explainable picture of memory pressure, context scaling, and concurrency constraints.

---

## The Problem

The AI ecosystem has excellent model leaderboards, benchmark trackers, and inference runtimes. What it lacks is tooling focused on infrastructure realism:

- Which AI workload can this GPU actually sustain?
- How does VRAM pressure evolve with context length?
- What happens when multiple users run concurrently?
- How does quantization change the picture?

Moreover.ai is built to answer those questions — not with fuzzy heuristics, but with verified data and a strict, auditable methodology.

---

## Core Philosophy: No Hallucinated Estimates

This is the most important design principle of the project.

Moreover.ai intentionally **refuses to produce estimates when the underlying data is missing**. This means:

- No fuzzy matching on model names
- No parameter guessing from strings like `"7B"` in a model name
- No invented context windows
- No inferred architecture values
- No synthetic benchmark scores

If a value cannot be sourced from an official, verifiable location — the field remains empty.

This is a deliberate choice. Infrastructure sizing tools must prioritize **transparency**, **reproducibility**, and **explainability** over the appearance of completeness. An empty field is honest. A fabricated estimate is dangerous.

---

## Enrichment Methodology

The enrichment pipeline follows a two-step process.

### Step 1 — Open LLM Leaderboard (Exact Match Only)

Models are matched against the [`open-llm-leaderboard/contents`](https://huggingface.co/datasets/open-llm-leaderboard/contents) dataset using **exact string matching** between the local `model_id` and the leaderboard `fullname`.

This step adds:
- Parameter counts
- Raw benchmark scores (IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO)

No fuzzy matching is used. If a model does not exactly match a leaderboard entry, nothing is assumed.

### Step 2 — Hugging Face Repository Metadata

The pipeline fetches official metadata directly from Hugging Face repositories:

- `config.json` — primary source for architecture fields
- `safetensors` metadata — used to derive parameter counts when not available from the leaderboard
- Repository metadata — supplementary source

The parser searches a curated list of **explicit configuration key aliases** (e.g. `hidden_size`, `n_embd`, `d_model` all map to the same field). No model-name inference will be performed at any point.

Fields extracted this way include:
- Architecture name and model type
- Hidden size, number of layers, attention heads
- KV heads (for GQA/MQA architectures)
- Intermediate (FFN) size
- Vocabulary size
- Context window (`max_position_embeddings` and its known aliases)
- Sliding window size, RoPE scaling, RoPE theta

### Step 3 — Memory and KV Cache Estimates

Once architecture fields are verified, the pipeline computes:

| Metric | Formula |
|---|---|
| Weight memory (FP16) | `parameters_b × 2 GB` |
| Weight memory (INT8) | `parameters_b × 1 GB` |
| Weight memory (INT4) | `parameters_b × 0.5 GB` |
| KV cache bytes/token | `2 × layers × kv_heads × head_dim × 2 bytes` |
| Total VRAM estimate | `(weights + kv_cache) × 1.2` (20% overhead) |

VRAM estimates are computed for three context scenarios: **4K**, **8K**, and **32K** tokens, and for each quantization level. Models missing any required architecture field produce no estimate — no partial or approximated values are emitted.

---

## Features

### Model Registry

- Open-weight and closed models tracked in a single registry
- Architecture metadata extracted from official sources
- Context window tracking
- Parameter sizing from leaderboard or repository metadata
- Raw benchmark scores (not normalized composites)

### GPU Infrastructure Sizing

For each model with sufficient metadata:

- FP16 / INT8 / INT4 weight memory footprint
- KV cache growth per token
- KV cache at 4K, 8K, 32K context
- Total estimated VRAM at each context × quantization combination

### AI Workload Presets

The sizing engine supports workload-oriented infrastructure planning. Define the workload type and the framework filters compatible models accordingly:

- Simple chatbot
- RAG over documents
- Large RAG
- Codebase analysis
- Agentic workflow
- Long-context analysis

### Infrastructure Heatmap

The interactive dashboard renders a GPU × context heatmap that visualizes:

- Memory pressure zones (comfortable / limited / insufficient)
- Context scaling impact
- Quantization impact
- GPU saturation across a fleet

Click any model to inspect its full architecture metadata, benchmark scores, and per-context VRAM breakdown.

---

## Repository Structure

```
.
├── enrich_models.py                       # Enrichment pipeline
├── index.html                             # Static interactive dashboard
├── models.json                            # Generated model registry
├── model-list.json                        # Input model list
├── gpu_registry.json                      # GPU fleet definitions
├── audit_no_estimate_open_weight_only.py  # Coverage audit tool
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/your-org/moreover-ai.git
cd moreover-ai
pip install -r requirements.txt
```

---

## Running the Enrichment Pipeline

```bash
python3 enrich_models.py
```

Output:
- `models.json` — enriched model registry used by the dashboard

---

## Running the Dashboard

```bash
python3 -m http.server 8000
```

Then open: [http://localhost:8000/index.html](http://localhost:8000/index.html)

---

## Audit Tool

The `audit_no_estimate_open_weight_only.py` script produces a full coverage report for open-weight models. It identifies which models could not receive a VRAM estimate and categorizes the reason:

- `missing_parameters_and_config` — no leaderboard match and no HF config
- `missing_parameters` — architecture found but parameter count unavailable
- `missing_config_for_kv_cache` — parameters known but architecture fields absent
- `closed_or_non_hf_model_no_config` — closed model, no public config
- `hf_collection_not_model_repo` — HF collection URL, not a model repo

```bash
python3 audit_no_estimate_open_weight_only.py
```

Outputs:
- `no_estimate_audit.xlsx` — per-model audit with missing fields
- `no_estimate_audit.json` — same data in JSON

---

## GPU Registry

The `gpu_registry.json` file defines the GPU fleet used by the heatmap. It covers consumer, workstation, datacenter, Apple Silicon, and AMD APU hardware. Each entry specifies:

```json
{
  "gpu_name": "RTX 4090",
  "memory_gb": 24,
  "vendor": "NVIDIA",
  "category": "Consumer"
}
```

Adding new hardware to the registry is as simple as appending entries to this file.

---

## Technical Stack

- **Python** — enrichment pipeline
- **Pandas** — data processing
- **Hugging Face Hub** — config and metadata fetching
- **Hugging Face Datasets** — Open LLM Leaderboard access
- **Vanilla HTML/JS** — static dashboard, no build step, no framework

---

## Roadmap

### Benchmarks & Real-World Scores
Leaderboard scores reflect academic benchmarks, not deployment realities. The next step is to complement them with usage-oriented metrics and community feedback — real-world performance data contributed by people running these models in production, on their own hardware, for actual workloads.

### LLM & GPU Ranking UX
The dashboard currently surfaces model metadata and memory estimates. The goal is to evolve it into a proper ranking interface — where models and GPUs can be compared, filtered, and ranked against each other based on infrastructure constraints, not just benchmark scores.

### Advanced Inference & Infrastructure Sizing
The sizing engine will expand to cover the full infrastructure picture: throughput and latency estimation, multi-GPU configurations, datacenter-scale deployments, and cost and power consumption estimates. The aim is to give teams everything they need to make a realistic infrastructure decision without leaving the tool.

---

## Contributing

Moreover.ai is intentionally open. Contributions are especially welcome from people working on:

- AI infrastructure and self-hosted AI
- Inference runtimes (vLLM, llama.cpp, TensorRT-LLM)
- GPU sizing and profiling
- Benchmark methodology
- Visualization and UX

Areas where contributions would have high impact:

- Additional architecture config key aliases for less common model families
- Runtime-specific memory overhead profiles
- Expanded GPU registry entries
- Improved context window detection for non-standard configs
- Dashboard UX and heatmap improvements

To contribute: fork the repository, make your changes, and open a pull request with a clear description of the methodology behind any new estimates.

---

## Disclaimer

Moreover.ai provides infrastructure estimates, not guarantees. Real-world inference performance depends on the runtime, batching strategy, CUDA kernel implementations, quantization quality, concurrent usage patterns, and hardware topology.

The goal of this project is to provide transparent, explainable, and reproducible sizing estimates — not benchmark marketing.

---

## License

See [LICENSE](./LICENSE) for details.
