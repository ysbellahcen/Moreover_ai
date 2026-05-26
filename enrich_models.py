import json
import math
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError

INPUT_FILE = "model-list.json"

OUTPUT_JSON_FILE = "models.json"

HF_DATASET = "open-llm-leaderboard/contents"
HF_SPLIT = "train"

LOCAL_MODEL_COL = "model_id"
HF_MODEL_COL = "fullname"

PARAMS_COL = "#Params (B)"

RAW_BENCHMARK_COLUMNS = [
    "IFEval Raw",
    "BBH Raw",
    "MATH Lvl 5 Raw",
    "GPQA Raw",
    "MUSR Raw",
    "MMLU-PRO Raw",
]

CONFIG_ALIASES = {
    "architectures": [
        "architectures",
    ],
    "model_type": [
        "model_type",
    ],
    "torch_dtype": [
        "torch_dtype",
        "dtype",
    ],
    "hidden_size": [
        "hidden_size",
        "n_embd",
        "d_model",
        "dim",
        "model_dim",
        "embedding_size",
        "embed_dim",
        "hidden_dim",
        "decoder_embed_dim",
    ],
    "num_hidden_layers": [
        "num_hidden_layers",
        "n_layer",
        "n_layers",
        "num_layers",
        "num_decoder_layers",
        "decoder_layers",
        "layers",
        "num_blocks",
        "num_hidden_layer",
    ],
    "num_attention_heads": [
        "num_attention_heads",
        "n_head",
        "n_heads",
        "num_heads",
        "attention_heads",
        "decoder_attention_heads",
        "num_query_heads",
    ],
    "num_key_value_heads": [
        "num_key_value_heads",
        "num_kv_heads",
        "n_kv_heads",
        "multi_query_group_num",
        "num_key_value_groups",
        "kv_heads",
        "num_kv_attention_heads",
        "multi_query_attention_heads",
    ],
    "intermediate_size": [
        "intermediate_size",
        "ffn_hidden_size",
        "n_inner",
        "mlp_hidden_size",
        "feed_forward_length",
        "decoder_ffn_dim",
    ],
    "vocab_size": [
        "vocab_size",
        "padded_vocab_size",
    ],
    "max_position_embeddings": [
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "max_sequence_length",
        "model_max_length",
        "max_seq_len",
        "max_seq_length",
        "max_context_length",
        "context_length",
    ],
    "sliding_window": [
        "sliding_window",
        "sliding_window_size",
        "attention_window",
    ],
    "rope_scaling": [
        "rope_scaling",
    ],
    "rope_theta": [
        "rope_theta",
        "rotary_emb_base",
    ],
}

CONFIG_KEYS_TO_EXPORT = list(CONFIG_ALIASES.keys())

CONTEXT_KEYS = CONFIG_ALIASES["max_position_embeddings"] + [
    "sliding_window",
    "sliding_window_size",
]

CONTEXT_SCENARIOS = [4096, 8192, 32768]

api = HfApi()


def normalize_param_value(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "" or text.lower() in ["nan", "none"]:
        return None

    try:
        number = float(text)
        if number.is_integer():
            return f"{int(number)}B"
        return f"{number:g}B"
    except Exception:
        return text


def parse_parameters_to_billions(value):
    """
    Converts only already-sourced values to numeric billions.
    No model-name inference.
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if text == "" or text.lower() in ["nan", "none"]:
        return None

    try:
        lower = text.lower().replace(",", "")

        if lower.endswith("b"):
            return float(lower[:-1])

        if lower.endswith("m"):
            return float(lower[:-1]) / 1000

        raw = float(lower)

        if raw > 1_000_000:
            return raw / 1_000_000_000

        return raw

    except Exception:
        return None


def is_probably_hf_repo(model_id):
    if not isinstance(model_id, str):
        return False

    model_id = model_id.strip()

    if "/" not in model_id:
        return False

    if model_id.startswith("collections/"):
        return False

    if model_id.count("/") != 1:
        return False

    return True


def read_hf_json(repo_id, filename):
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (RepositoryNotFoundError, EntryNotFoundError, HfHubHTTPError):
        return None
    except Exception:
        return None


def search_nested(obj, keys, visited=None):
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] is not None and obj[key] != "":
                return obj[key]
        for value in obj.values():
            result = search_nested(value, keys, visited)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = search_nested(item, keys, visited)
            if result is not None:
                return result

    return None


def extract_config_fields(config):
    extracted = {}

    for field, aliases in CONFIG_ALIASES.items():
        value = search_nested(config, aliases)
        extracted[field] = value

    return extracted


def extract_context_from_config(config, extracted=None):
    if extracted is None:
        extracted = extract_config_fields(config)

    context = extracted.get("max_position_embeddings")

    if context is not None:
        try:
            return int(context)
        except Exception:
            pass

    sliding = extracted.get("sliding_window")
    if sliding is not None:
        try:
            return int(sliding)
        except Exception:
            pass

    return None


def extract_parameters_from_repo(repo_id, config=None):
    try:
        info = api.model_info(repo_id=repo_id)

        if hasattr(info, "safetensors") and info.safetensors is not None:
            st = info.safetensors
            if hasattr(st, "total") and st.total is not None:
                params_raw = st.total
                params_b = params_raw / 1_000_000_000
                return f"{params_b:g}B"

        if hasattr(info, "cardData") and info.cardData:
            card = info.cardData
            for key in ["model-index", "model_parameters", "num_parameters"]:
                if key in card and card[key] is not None:
                    try:
                        raw = float(str(card[key]).replace(",", ""))
                        if raw > 1_000_000:
                            raw = raw / 1_000_000_000
                        return f"{raw:g}B"
                    except Exception:
                        pass

    except Exception:
        pass

    if config is not None:
        for key in ["num_parameters", "n_parameters", "total_params"]:
            if key in config and config[key] is not None:
                try:
                    raw = float(config[key])
                    if raw > 1_000_000:
                        raw = raw / 1_000_000_000
                    return f"{raw:g}B"
                except Exception:
                    pass

    return None


def read_hf_config(repo_id):
    config = read_hf_json(repo_id, "config.json")

    if config is None:
        for subfolder in ["src", "model"]:
            config = read_hf_json(repo_id, f"{subfolder}/config.json")
            if config is not None:
                break

    return config


def to_float_or_none(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)

    try:
        return float(value)
    except Exception:
        return None


def calculate_weight_memory(parameters_b):
    if parameters_b is None:
        return None, None, None

    fp16 = parameters_b * 2
    int8 = parameters_b * 1
    int4 = parameters_b * 0.5

    return round(fp16, 3), round(int8, 3), round(int4, 3)


def calculate_kv_cache_bytes_per_token(row, kv_bytes=2):
    hidden_size = to_float_or_none(row.get("hidden_size"))
    num_layers = to_float_or_none(row.get("num_hidden_layers"))
    num_attention_heads = to_float_or_none(row.get("num_attention_heads"))
    num_key_value_heads = to_float_or_none(row.get("num_key_value_heads"))

    # Explicit config convention: if kv heads absent, classic multi-head attention has kv heads = attention heads.
    # This is not inferred from model name; it is a standard attention structure assumption.
    if num_key_value_heads is None and num_attention_heads is not None:
        num_key_value_heads = num_attention_heads

    if not hidden_size or not num_layers or not num_attention_heads or not num_key_value_heads:
        return None

    head_dim = hidden_size / num_attention_heads

    bytes_per_token = 2 * num_layers * num_key_value_heads * head_dim * kv_bytes

    return round(bytes_per_token, 3)


def bytes_to_gb(value):
    if value is None:
        return None

    return value / 1_000_000_000


def calculate_vram_estimates(row):
    params_b = parse_parameters_to_billions(row.get("parameters"))

    weight_fp16, weight_int8, weight_int4 = calculate_weight_memory(params_b)

    kv_bytes_per_token_fp16 = calculate_kv_cache_bytes_per_token(row, kv_bytes=2)
    kv_mb_per_token_fp16 = (
        round(kv_bytes_per_token_fp16 / 1_000_000, 6)
        if kv_bytes_per_token_fp16 is not None
        else None
    )

    estimates = {
        "parameters_b": params_b,
        "weight_memory_fp16_gb": weight_fp16,
        "weight_memory_int8_gb": weight_int8,
        "weight_memory_int4_gb": weight_int4,
        "kv_cache_bytes_per_token_fp16": kv_bytes_per_token_fp16,
        "kv_cache_mb_per_token_fp16": kv_mb_per_token_fp16,
    }

    for context_tokens in CONTEXT_SCENARIOS:
        kv_gb = (
            bytes_to_gb(kv_bytes_per_token_fp16 * context_tokens)
            if kv_bytes_per_token_fp16 is not None
            else None
        )

        estimates[f"kv_cache_fp16_context_{context_tokens}_gb"] = (
            round(kv_gb, 3) if kv_gb is not None else None
        )

        for quant, weight_memory in [
            ("fp16", weight_fp16),
            ("int8", weight_int8),
            ("int4", weight_int4),
        ]:
            if weight_memory is None or kv_gb is None:
                estimates[f"estimated_vram_{quant}_context_{context_tokens}_gb"] = None
            else:
                total = (weight_memory + kv_gb) * 1.2
                estimates[f"estimated_vram_{quant}_context_{context_tokens}_gb"] = round(total, 3)

    return estimates


def export_json(df, output_file):
    clean_df = df.copy()
    clean_df = clean_df.astype(object)
    clean_df = clean_df.where(pd.notnull(clean_df), None)

    records = clean_df.to_dict(orient="records")

    def clean_value(value):
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, dict):
            return {k: clean_value(v) for k, v in value.items()}

        if isinstance(value, list):
            return [clean_value(v) for v in value]

        return value

    records = [clean_value(record) for record in records]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, allow_nan=False)

    return len(records)


print("Loading local JSON...")
df = pd.read_json(INPUT_FILE)

if LOCAL_MODEL_COL not in df.columns:
    raise ValueError(f"Missing column in local JSON: {LOCAL_MODEL_COL}")

df[LOCAL_MODEL_COL] = df[LOCAL_MODEL_COL].astype(str).str.strip()

print("Loading Hugging Face leaderboard dataset...")
ds = load_dataset(HF_DATASET, split=HF_SPLIT)
hf = ds.to_pandas()

if HF_MODEL_COL not in hf.columns:
    raise ValueError(f"Missing column in HF dataset: {HF_MODEL_COL}")

available_columns = [HF_MODEL_COL]

if PARAMS_COL in hf.columns:
    available_columns.append(PARAMS_COL)
else:
    print(f"Warning: '{PARAMS_COL}' not found. Parameters from leaderboard will be empty.")

available_raw_benchmarks = [
    col for col in RAW_BENCHMARK_COLUMNS
    if col in hf.columns
]

available_columns.extend(available_raw_benchmarks)

print("\nRaw benchmark columns used from leaderboard:")
for col in available_raw_benchmarks:
    print("-", col)

hf_small = hf[available_columns].copy()
hf_small[HF_MODEL_COL] = hf_small[HF_MODEL_COL].astype(str).str.strip()

hf_small = hf_small.rename(columns={HF_MODEL_COL: LOCAL_MODEL_COL})

if PARAMS_COL in hf_small.columns:
    hf_small["parameters"] = hf_small[PARAMS_COL].apply(normalize_param_value)
    hf_small = hf_small.drop(columns=[PARAMS_COL])
else:
    hf_small["parameters"] = None

rename_raw_benchmarks = {
    "IFEval Raw": "benchmark_ifeval_raw",
    "BBH Raw": "benchmark_bbh_raw",
    "MATH Lvl 5 Raw": "benchmark_math_lvl_5_raw",
    "GPQA Raw": "benchmark_gpqa_raw",
    "MUSR Raw": "benchmark_musr_raw",
    "MMLU-PRO Raw": "benchmark_mmlu_pro_raw",
}

hf_small = hf_small.rename(columns={
    old: new
    for old, new in rename_raw_benchmarks.items()
    if old in hf_small.columns
})

hf_small = hf_small.drop_duplicates(subset=[LOCAL_MODEL_COL])

print("\nStep 1 — exact match with Open LLM Leaderboard...")
merged = df.merge(
    hf_small,
    on=LOCAL_MODEL_COL,
    how="left"
)

if "parameters" not in merged.columns:
    merged["parameters"] = None

if "context" not in merged.columns:
    merged["context"] = None

for key in CONFIG_KEYS_TO_EXPORT:
    if key not in merged.columns:
        merged[key] = None

print("\nStep 2 — official Hugging Face repo recursive config + metadata enrichment...")

repo_checked = 0
params_filled_repo = 0
context_filled_repo = 0
configs_found = 0
config_fields_filled = 0

for idx, row in merged.iterrows():
    repo_id = row[LOCAL_MODEL_COL]

    if not is_probably_hf_repo(repo_id):
        continue

    needs_params = pd.isna(row.get("parameters"))
    needs_context = pd.isna(row.get("context"))
    needs_config = any(pd.isna(row.get(key)) for key in CONFIG_KEYS_TO_EXPORT)

    if not needs_params and not needs_context and not needs_config:
        continue

    repo_checked += 1

    config = read_hf_config(repo_id)

    if config is not None:
        configs_found += 1

        extracted = extract_config_fields(config)

        for key, value in extracted.items():
            if value is not None and value != "" and pd.isna(merged.at[idx, key]):
                merged.at[idx, key] = value
                config_fields_filled += 1

        if needs_context:
            context = extract_context_from_config(config, extracted)

            if context is not None:
                merged.at[idx, "context"] = context
                context_filled_repo += 1

    if needs_params:
        repo_params = extract_parameters_from_repo(repo_id, config=config)

        if repo_params is not None:
            merged.at[idx, "parameters"] = repo_params
            params_filled_repo += 1

print("\nStep 3 — memory and KV cache estimates from verified fields...")

estimate_rows = []

for _, row in merged.iterrows():
    estimate_rows.append(calculate_vram_estimates(row))

estimate_df = pd.DataFrame(estimate_rows)

for col in estimate_df.columns:
    if col in merged.columns:
        merged = merged.drop(columns=[col])

merged = pd.concat([merged.reset_index(drop=True), estimate_df.reset_index(drop=True)], axis=1)

total = len(merged)
matched_params = merged["parameters"].notna().sum()
matched_context = merged["context"].notna().sum()

benchmark_cols_out = [
    col for col in merged.columns
    if col.startswith("benchmark_") and col.endswith("_raw")
]

benchmark_matches = (
    merged[benchmark_cols_out].notna().any(axis=1).sum()
    if benchmark_cols_out
    else 0
)

kv_ready = merged["kv_cache_bytes_per_token_fp16"].notna().sum()

print(f"\nRows in local JSON: {total}")
print(f"Rows with parameters: {matched_params}/{total}")
print(f"Rows with context: {matched_context}/{total}")
print(f"Rows with at least one raw benchmark score: {benchmark_matches}/{total}")
print(f"HF repos checked in step 2: {repo_checked}")
print(f"Configs found: {configs_found}")
print(f"Config fields filled: {config_fields_filled}")
print(f"Parameters filled from repo metadata: {params_filled_repo}")
print(f"Context filled from config.json: {context_filled_repo}")
print(f"Rows ready for KV cache calculation: {kv_ready}/{total}")

print("\nSaving JSON output...")
json_rows = export_json(merged, OUTPUT_JSON_FILE)
print(f"JSON done: {OUTPUT_JSON_FILE} ({json_rows} records)")

print("\nDone.")
