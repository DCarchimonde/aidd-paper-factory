# Paper 2 RACER-C Phase 3 RTX-4060 benchmark result review

Date: 2026-08-06
Status: **seed-99 engineering gate passed; protocol remains pre-freeze**

## Decision

The corrected Windows RTX-4060 component benchmark passed. Seed 99 remains
development-only and is excluded from every scientific comparison. No performance
metric was computed and no policy, conformal, or test prediction was generated.
Confirmatory seeds 101--110 remain blocked.

## Audited result

The local result reports:

- endpoint/track: `Tox21_NR_ER` / strict scaffold;
- source/model-eligible/development counts: 5,855 / 5,852 / 2,926;
- transparent overlength exclusions: 3 structures at 227, 239, and 242 tokens;
- MoLFormer: 2,926 embeddings in 7.0394 s, 0.218 GiB peak allocated and
  0.266 GiB peak reserved;
- Chemprop: 1,755 fit rows, 195 internal-validation rows, 976 predictions;
- Chemprop train/predict time: 30.5265 / 14.2483 s;
- Chemprop train/predict peak GPU memory: 2,376 / 2,194 MiB;
- finite probabilities and lineage records: 976 / 976;
- environment audit: Windows amd64, Python 3.11.13, exact candidate package
  versions, CUDA 13.0, and NVIDIA GeForce RTX 4060 Laptop GPU;
- measured primary D-MPNN projection: 3.0527 GPU-hours, or 3.6632 GPU-hours
  with the predeclared 20% rerun reserve.

The timing projection is an engineering estimate, not a scientific outcome. It
replaces the historical 150--400 RTX-4090-hour planning range for the 60 primary
endpoint/track/seed cells only. CPU work, anchor sensitivities, external-method
baselines, and implementation-debug time are not silently included in 3.6632 h.

## Hash and lineage contract

The result records the clean and role-input hashes, MoLFormer model/tokenizer
revision, eligible-cohort and token-contract hashes, environment `pip freeze`
hash, prediction-file hash, and the byte hashes of the locally checked-out config
and benchmark runner. The formal review script compares the latter two against
the same Windows checkout; this avoids treating LF/CRLF byte differences as a
scientific change while still failing closed on a changed local runner.

## Remaining freeze gate

The benchmark audited model-domain eligibility only for NR-ER. Before the four
primary endpoints can be frozen, the same label-blind 202-token rule must be
applied to NR-AhR, NR-ER, SR-ARE, and SR-MMP before role assignment, followed by
the complete 4 endpoints × 3 tracks × 5 seeds count audit. The prediction-free
`FreezeReview` mode performs this as one operation and writes one review JSON.
Only after that record passes, the production implementation and protocol text
pass their tests, and the user explicitly approves may the freeze tag be created.
