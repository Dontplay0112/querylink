<h1 align="center">QueryLink: Leveraging Query-Memory Alignment for Long-Term Reasoning in LLM Agents</h1>

This repository contains the official implementation of **QueryLink**, a framework that bridges the semantic gap between queries and long-term memories through dual-side, multi-view alignment and coherent memory chunking.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key when using the default LLM configuration
- A CUDA-capable GPU is recommended for local embedding models

## Installation

```bash
git clone https://github.com/Dontplay0112/querylink.git
cd querylink
uv sync --frozen
cp .env.example .env
```

Add the credentials required by your selected model provider to `.env`. The default configuration reads `OPENAI_API_KEY`.

Embedding models are downloaded automatically on first use. Download the NLTK and spaCy resources used by the evaluation code with:

```bash
uv run python src/prepare.py
```

## Datasets

Datasets are not redistributed in this repository. Place them under `data/` using the following layout:

```text
data/
├── locomo/
│   ├── locomo10.json
│   └── locomo10.pkl
└── longmemeval/
    └── longmemeval_s.json
```

Obtain each dataset from its official source and follow its license and terms. In particular, the [LoCoMo repository](https://github.com/snap-research/locomo) is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); the dataset is not covered by QueryLink's Apache-2.0 license.

## Reproduction

Run an experiment from the repository root:

```bash
# LoCoMo
bash run.sh locomo

# LongMemEval
bash run.sh longmemeval
```

The launcher no longer hard-codes GPU IDs. To restrict which devices the runtime may select, set `CUDA_VISIBLE_DEVICES` explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh locomo
```

Results and configuration snapshots are written under `outputs/`.

Run the lightweight configuration regression tests with:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

## Configuration

Experiment configurations are stored in `config/`. Relative data and output paths are resolved from the repository root.

```toml
[exp]
task = "locomo"
desc = "reproduction"
base_data_path = "data/"
data_file_name = "locomo10.json"
base_output_path = "outputs/"
max_workers = 10
max_inflight = 10
max_thread_workers = 16
max_thread_inflight = 16

[agent]
agent = "QueryLink"
llm_model_source = "openai"
llm_model_name = "gpt-4o-mini"
llm_temperature = 0.0
embed_model_source = "huggingface"
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

The main options are:

| Parameter | Description |
| --- | --- |
| `task` | Dataset/task adapter: `locomo` or `longmemeval`. |
| `base_data_path` | Dataset root, relative to the repository root. |
| `data_file_name` | Input dataset filename. |
| `base_output_path` | Output root, relative to the repository root. |
| `max_workers` / `max_inflight` | Process-level concurrency limits. |
| `max_thread_workers` / `max_thread_inflight` | Thread-level concurrency limits. |
| `agent` | Agent implementation registered in `src/agent/__init__.py`. |
| `llm_model_source` / `llm_model_name` | LLM provider and model name. |
| `llm_temperature` | LLM sampling temperature. |
| `embed_model_source` / `embed_model_name` | Embedding provider and model name. |
| `memory_version` | Optional existing memory snapshot to reuse. |

Unknown configuration keys are rejected so that misspelled or obsolete options do not silently fall back to defaults.

## License and third-party software

Original QueryLink code is released under the [Apache License 2.0](LICENSE). Portions adapted from A-Mem remain subject to its MIT license and attribution requirements, and the evaluation prompt adapted from Mem0 remains under Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

The datasets and model weights used by the experiments are distributed separately under their respective terms.

## Citation

If you use QueryLink in your research, please cite:

```bibtex
@inproceedings{hu2026querylink,
  title={QueryLink: Leveraging Query-Memory Alignment for Long-Term Reasoning in LLM Agents},
  author={Hu, Xuxian and Teng, Zhu and Zhang, Wei and He, Ming and Fan, Jianping},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={15608--15621},
  year={2026}
}
```
