from __future__ import annotations

"""Label-blind MoLFormer token-domain eligibility for RACER-C.

The pinned MoLFormer model card states that molecules longer than 202 tokens
were removed during pretraining.  RACER-C therefore excludes such structures
from every predictor block before role allocation; it never truncates a SMILES
or extends the pretrained positional domain.
"""

import hashlib
import json
import re
from typing import Iterable, Mapping


MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"
TOKENIZER_JSON_SHA256 = (
    "3df1f2219653c44fac9fa03b7f788b372eb2544ecc176737bb9aca8411b471a5"
)
MAX_TOKENS_INCLUDING_SPECIAL_TOKENS = 202
OVERLENGTH_ACTION = "exclude_before_role_assignment_and_all_component_fits"
INPUT_COLUMN = "standardized_smiles"

# Exact Split regex in tokenizer.json at the immutable revision above.
TOKEN_PATTERN_TEXT = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|"
    r"\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
TOKEN_PATTERN = re.compile(TOKEN_PATTERN_TEXT)


def token_count_including_special_tokens(smiles: str) -> int:
    tokens = TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError("SMILES contains text not covered by the pinned tokenizer regex")
    return len(tokens) + 2  # <bos>, molecular tokens, <eos>


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_config(config: Mapping[str, object]) -> Mapping[str, object]:
    block = config["molformer"]
    expected = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "input_column": INPUT_COLUMN,
        "max_tokens_including_special_tokens": MAX_TOKENS_INCLUDING_SPECIAL_TOKENS,
        "truncation": False,
        "overlength_action": OVERLENGTH_ACTION,
    }
    failures = [
        f"{key}: expected {value!r}, observed {block.get(key)!r}"
        for key, value in expected.items()
        if block.get(key) != value
    ]
    if failures:
        raise RuntimeError("MoLFormer token contract mismatch: " + "; ".join(failures))
    return block


def filter_model_eligible_rows(
    role_rows: Iterable[Mapping[str, str]],
    clean_rows: Iterable[Mapping[str, str]],
    config: Mapping[str, object],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    block = validate_config(config)
    materialized_role = [dict(row) for row in role_rows]
    materialized_clean = [dict(row) for row in clean_rows]
    role_ids = [row["structure_id"] for row in materialized_role]
    clean_ids = [row["structure_id"] for row in materialized_clean]
    if len(role_ids) != len(set(role_ids)) or len(clean_ids) != len(set(clean_ids)):
        raise ValueError("model eligibility requires unique structure IDs")
    if set(role_ids) != set(clean_ids):
        raise ValueError("role and clean rows differ before model eligibility")

    lengths: dict[str, int] = {}
    exclusions: list[dict[str, object]] = []
    for row in materialized_clean:
        structure_id = row["structure_id"]
        smiles = row[str(block["input_column"])]
        length = token_count_including_special_tokens(smiles)
        lengths[structure_id] = length
        if length > int(block["max_tokens_including_special_tokens"]):
            exclusions.append(
                {
                    "structure_id": structure_id,
                    "source_record_id": row.get("source_record_id", ""),
                    "token_count_including_special_tokens": length,
                    "standardized_smiles_sha256": hashlib.sha256(
                        smiles.encode("utf-8")
                    ).hexdigest(),
                }
            )

    exclusions.sort(key=lambda row: str(row["structure_id"]))
    excluded_ids = {str(row["structure_id"]) for row in exclusions}
    eligible_role = [
        row for row in materialized_role if row["structure_id"] not in excluded_ids
    ]
    eligible_clean = [
        row for row in materialized_clean if row["structure_id"] not in excluded_ids
    ]
    eligible_lengths = {
        row["structure_id"]: lengths[row["structure_id"]] for row in eligible_clean
    }
    if not eligible_clean:
        raise ValueError("MoLFormer eligibility removed the entire endpoint")
    if max(eligible_lengths.values()) > int(block["max_tokens_including_special_tokens"]):
        raise AssertionError("overlength molecule survived model eligibility")

    contract = {
        "model_id": block["model_id"],
        "revision": block["revision"],
        "tokenizer_json_sha256": block["tokenizer_json_sha256"],
        "input_column": block["input_column"],
        "max_tokens_including_special_tokens": block[
            "max_tokens_including_special_tokens"
        ],
        "truncation": block["truncation"],
        "overlength_action": block["overlength_action"],
    }
    report = {
        "contract": contract,
        "contract_sha256": _stable_sha256(contract),
        "source_n": len(materialized_clean),
        "eligible_n": len(eligible_clean),
        "excluded_n": len(exclusions),
        "source_max_tokens_observed": max(lengths.values()),
        "eligible_max_tokens_observed": max(eligible_lengths.values()),
        "excluded": exclusions,
        "eligible_cohort_sha256": _stable_sha256(
            sorted(eligible_lengths.items())
        ),
        "selection_uses_labels": False,
    }
    return eligible_role, eligible_clean, report


def verify_runtime_tokenizer(
    clean_rows: Iterable[Mapping[str, str]], tokenizer: object, input_column: str
) -> None:
    rows = [dict(row) for row in clean_rows]
    for start in range(0, len(rows), 512):
        batch = rows[start : start + 512]
        smiles = [row[input_column] for row in batch]
        encoded = tokenizer(
            smiles,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        observed = [len(value) for value in input_ids]
        expected = [token_count_including_special_tokens(value) for value in smiles]
        if observed != expected:
            index = next(i for i, pair in enumerate(zip(observed, expected)) if pair[0] != pair[1])
            raise RuntimeError(
                "runtime tokenizer differs from pinned offline token contract for "
                f"structure_id={batch[index]['structure_id']}: "
                f"runtime={observed[index]} expected={expected[index]}"
            )
