import json
import math
from collections import Counter
import pandas as pd

INPUT_JSON = "models.json"
OUTPUT_EXCEL = "no_estimate_audit.xlsx"
OUTPUT_JSON = "no_estimate_audit.json"

QUANTIZATIONS = ["fp16", "int8", "int4"]

def is_empty(value):
    if value is None:
        return True

    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)

    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in ["nan", "none", "null"]

    return False

def has_any_weight_memory(row):
    return any(
        not is_empty(row.get(f"weight_memory_{q}_gb"))
        for q in QUANTIZATIONS
    )

def detect_missing_fields(row):
    missing = []

    # Weight memory comes from parameters.
    if is_empty(row.get("parameters")):
        missing.append("parameters")
    if is_empty(row.get("parameters_b")):
        missing.append("parameters_b")
    if not has_any_weight_memory(row):
        missing.append("weight_memory")

    # KV cache requires architecture/config fields.
    if is_empty(row.get("kv_cache_bytes_per_token_fp16")):
        missing.append("kv_cache_bytes_per_token_fp16")

    for key in [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
    ]:
        if is_empty(row.get(key)):
            missing.append(key)

    # num_key_value_heads can be absent for classic MHA models because the dashboard can fallback
    # to num_attention_heads, so we treat it as informative, not always blocking.
    if is_empty(row.get("num_key_value_heads")):
        missing.append("num_key_value_heads_optional")

    return missing

def reason_from_missing(row, missing):
    model_id = str(row.get("model_id", ""))

    if "/" not in model_id:
        return "closed_or_non_hf_model_no_config"
    if model_id.startswith("collections/"):
        return "hf_collection_not_model_repo"
    if "parameters" in missing and "kv_cache_bytes_per_token_fp16" in missing:
        return "missing_parameters_and_config"
    if "parameters" in missing:
        return "missing_parameters"
    if "kv_cache_bytes_per_token_fp16" in missing:
        return "missing_config_for_kv_cache"
    if "weight_memory" in missing:
        return "missing_weight_memory"
    return "other"

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)

    audit_rows = []

    for row in rows:
        missing = detect_missing_fields(row)

        no_estimate = (
            not has_any_weight_memory(row)
            or is_empty(row.get("kv_cache_bytes_per_token_fp16"))
        )

        # V7 focus:
        # keep only open-weight models
        if row.get("open_weight") is not True:
            continue

        if no_estimate:
            audit_rows.append({
                "model_id": row.get("model_id"),
                "display_name": row.get("display_name"),
                "provider": row.get("provider"),
                "family": row.get("family"),
                "open_weight": row.get("open_weight"),
                "parameters": row.get("parameters"),
                "context": row.get("context"),
                "model_type": row.get("model_type"),
                "hidden_size": row.get("hidden_size"),
                "num_hidden_layers": row.get("num_hidden_layers"),
                "num_attention_heads": row.get("num_attention_heads"),
                "num_key_value_heads": row.get("num_key_value_heads"),
                "kv_cache_bytes_per_token_fp16": row.get("kv_cache_bytes_per_token_fp16"),
                "weight_memory_fp16_gb": row.get("weight_memory_fp16_gb"),
                "weight_memory_int8_gb": row.get("weight_memory_int8_gb"),
                "weight_memory_int4_gb": row.get("weight_memory_int4_gb"),
                "missing_fields": ", ".join(missing),
                "main_reason": reason_from_missing(row, missing),
            })

    df = pd.DataFrame(audit_rows)

    summary = Counter(df["main_reason"]) if not df.empty else Counter()

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="no_estimate_models", index=False)

        summary_df = pd.DataFrame([
            {"reason": reason, "count": count}
            for reason, count in summary.items()
        ]).sort_values("count", ascending=False)

        summary_df.to_excel(writer, sheet_name="summary", index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_rows, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"Total models: {len(rows)}")
    print(f"No estimate models: {len(audit_rows)}")
    print("\nReasons:")
    for reason, count in summary.most_common():
        print(f"- {reason}: {count}")

    print(f"\nDone: {OUTPUT_EXCEL}")
    print(f"Done: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
