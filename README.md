<!-- LOCOMO需要使用mylocomoagent

需要手动装torch(可能)

tomllib在Python 3.11及以上版本自带，低版本需要安装tomli

all the path are relative to the src/ folder. -->

<p align="center">
    <h1 align="center">QueryLink: Leveraging Query-Memory Alignment for Long-Term Reasoning in LLM Agents</h1>
</p>

QueryLink bridges the semantic gap in agentic long-term memory via dual-side multi-view alignment, enabling robust reasoning and retrieval performance within a highly efficient flat memory architecture.

*The code will be released after the conference. Any questions can be raised via Issues or contacting the corresponding author.*

## Quickstart Guide

### Python Environment Setup

Configure your Python environment with uv (recommended):

```bash
# that's all.
uv sync
```

### Datasets

Datasets should be placed in the `data/` directory. File structure should be as follows:

```
data/
├── locomo/
│   ├── locomo10.json
│   ├── locomo10.pkl
├── longmemeval/
│   ├── longmemeval_s.json
```

### LLMS, Embeddings and other dependencies

**LLMs and Embeddings**

1) To use the default OpenAI models, set up your OpenAI API key in `.env` file.

2) Embedding models will be automatically downloaded when you run the code for the first time.

**nltk**

To evaluate the f1 score and bleu score, you need to install nltk and download the relevant resources, we prepare a simple python script to do that:

```bash
uv run src/prepare.py
```

## Reproduction

```bash
# for locomo
sh run.sh locomo
# for longmemeval
sh run.sh longmemeval
```

### Config File Explanation

Config files look like this:

```toml
# all the paths are relative to src/
[exp]
task = "locomo"
desc = "reproduction"
base_data_path = "../data/"
data_file_name = "locomo10.json"
base_output_path = "../outputs/"
max_workers = 10
max_inflight = 10
max_thread_workers = 16
max_thread_inflight = 16
# date = "20260217-105952"

[agent]
agent = "QueryLink"
model_source = "openai" # openai / huggingface / ollama
model_name = "gpt-4o-mini" # gpt-4o-mini / llama3.2:3b / qwen2.5:1.5b
temperature = 0.0
embed_model_source = "huggingface" # openai / huggingface / ollama
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # thenlper/gte-large NovaSearch/stella_en_1.5B_v5 facebook/contriever  mahonzhan/all-MiniLM-L6-v2

# memory_version = "20251213-171802-1-1-1"
```

Every time you run the code, the config file will be automatically backed up to the output directory, and the backup config file will be appended with the current date and time. You can modify the parameters in the config file to run different experiments. Below is an explanation of each parameter:

| Parameter | Description |
| --- | --- |
| `task` | The name of the task, determines the dataset directory relative to `base_data_path` and the output directory relative to `base_output_path`, it alse determines the way to read data. |
| `desc` | A brief description of the experiment content, only be saved in backup config file. |
| `base_data_path` | The base directory where datasets are stored. |
| `data_file_name` | The name of the dataset file to be read in the experiment. |
| `base_output_path` | The base directory where outputs will be saved. |
| `max_workers` | The maximum number of worker processes for parallel execution. |
| `max_inflight` | The maximum number of tasks that can be in flight at the same time(in sample level, for example, a `conversation` which contains multiple `sessions` in locomo dataset). |
| `max_thread_workers` | The maximum number of worker threads for parallel execution. |
| `max_thread_inflight` | The maximum number of tasks that can be in flight at the same time for threads(in memory unit level and qa pair level) . |
| `date` (Optional) | To continue a previous experiment when the code is unexpectedly interrupted, you can specify the `date` parameter, and the code will automatically find the backup config file corresponding to the date and load the parameters in it to continue the experiment. |
| `agent` | The name of the agent, determines the way to process data, build memory and do reasoning. |
| `model_source` | The provider of the language model (e.g., "openai", "huggingface", "ollama"). |
| `model_name` | The name of the language model to be used (e.g., "gpt-4o-mini", "llama3.2:3b", "qwen2.5:1.5b"). |
| `temperature` | The temperature setting for the language model, controlling the randomness of the output. |
| `embed_model_source` | The provider of the embedding model (e.g., "openai", "huggingface", "ollama"). |
| `embed_model_name` | The name of the embedding model to be used (e.g., "sentence-transformers/all-MiniLM-L6-v2", "thenlper/gte-large", "NovaSearch/stella_en_1.5B_v5", "facebook/contriever", "mahonzhan/all-MiniLM-L6-v2"(ollama)). |
| `memory_version` (Optional) | To skip the memory building process and directly load the memory which is built in past experiment, you can specify the version of the memory to be loaded. The code will automatically find the memory file in the output directory according to the version you specified. |

## Citation

Coming soon.

## License

Apache 2.0