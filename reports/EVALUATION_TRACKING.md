# NeuroDetect: F1 tracking, branding, and model verification

The product name remains **NeuroDetect**. The selected visual direction is dark
navy with restrained purple accents. Existing layouts and upload/chat interactions
remain; shared palette tokens live in `frontend/styles/palette.css`. Both partial
HTML and the generated index carry the restored name. The initial pastel draft
and Nurcutio branding were replaced following the user's correction.

## Verified model architecture

The actual `brain_tumor_multiclass.keras` archive contains a nested Functional
backbone named `mobilenetv2_1.00_224`: MobileNetV2 with width multiplier 1.0 and
224x224 RGB input. It has 154 backbone layers, including 17 DepthwiseConv2D layers.
The custom head is GlobalAveragePooling2D, Dense(256, ReLU), Dropout(0.4),
Dense(128, ReLU), Dropout(0.3), Dense(4, softmax). Class order remains glioma,
meningioma, notumor, pituitary. No trained weight file was modified. This verifies
architecture, not clinical validity or the provenance of training weights.

## Full labelled evaluation

The 11 supplied examples remain unlabelled regression cases and are excluded from
F1 estimates. Every image in the four existing Kaggle test directories was evaluated:
**1,311 images**. Existing directory labels were unchanged. Source independence from
training and population representativeness have not been externally audited, so these
are dataset-specific metrics rather than clinical performance estimates.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| glioma | 0.9748 | 0.7733 | 0.8625 | 300 |
| meningioma | 0.7902 | 0.7876 | 0.7889 | 306 |
| notumor | 0.9199 | 0.9926 | 0.9549 | 405 |
| pituitary | 0.8852 | 0.9767 | 0.9287 | 300 |

Macro-F1: **0.8837**; weighted-F1: **0.8890**; accuracy: **89.09%**.

Confusion matrix: rows=true labels, columns=predictions.

| Actual / predicted | glioma | meningioma | notumor | pituitary |
|---|---:|---:|---:|---:|
| glioma | 232 | 58 | 3 | 7 |
| meningioma | 5 | 241 | 29 | 31 |
| notumor | 0 | 3 | 402 | 0 |
| pituitary | 1 | 3 | 3 | 293 |

The weakest class here is meningioma; glioma recall also trails the other classes.
The confusion matrix makes these failure patterns visible instead of masking them
behind overall accuracy. All labels and misclassifications remain in the saved JSON.

A second complete run evaluated the same 1,311 files through the running local HTTP
API using the same workflow used for deployed evaluation. It produced **zero changes**
in accuracy, macro-F1, weighted-F1 and every per-class F1. This is repeatability evidence
for local serving, not evidence that production monitoring is already operating.

## Saved records and comparisons

- `reports/evaluations/baseline.json`: fixed original-model baseline.
- `reports/evaluations/<run-id>.json` and matching Markdown: immutable local run.
- `reports/evaluations/latest.md`: readable local metrics and confusion matrix.
- `reports/serving-evaluations/<run-id>.json`: second, complete black-box local API run.
- `reports/serving-evaluations/latest.md`: metrics and baseline deltas.

Records include per-image truth/prediction/confidence and SHA-256, dataset fingerprint,
class order, model SHA-256, version, source commit/dirty state, timestamp and duration.
`compare_results` rejects changed datasets/class orders. The deployed evaluator checks
the model hash before and after the run and refuses to save a result if it changes.
Undefined precision/recall/F1 divisions use zero; missing classes are flagged explicitly.

## Repeatable post-deployment evaluation

`.github/workflows/evaluate.yml` provides successful deployment-status, manual-dispatch,
and weekly Monday triggers. It evaluates the actual configured API, compares against
the committed baseline, publishes a run summary and uploads timestamped artifacts with
90-day retention. It does not load TensorFlow in CI or require an LLM API key.

Activation steps:

1. Push the workflow and baseline to the repository default branch.
2. Deploy the updated backend so `/api/ready` exposes its model SHA-256.
3. Set repository variable `EVALUATION_API_URL` to that backend's base URL.
4. Ensure Render deployments emit GitHub deployment status events if an immediate
   run is required after every deployment. This event bridge is not configured here.
   Manual and weekly runs work independently once the workflow and API variable exist.

The workflow is prepared locally and is **not claimed active or deployed**. It is an
explicit labelled-dataset evaluation, not artificial keep-alive polling. A model-version
label is supplied by the job and the actual serving weight hash is recorded separately.
CLI `--compare` accepts any archived run for cross-version comparisons on the same data.
No degradation threshold is silently used to approve/reject a model: the report records
deltas for review. Training/test overlap and real-world drift need separate auditing.

## Verification

- Full labelled evaluation: 1,311/1,311 completed locally.
- Black-box serving evaluation: 1,311/1,311 completed; all aggregate deltas zero.
- Python with integration: **58 passed, 7 deliberately skipped, zero failures**.
- Lightweight Python invocation: **33 passed, 32 deliberately skipped**.
- Frontend regression tests: **3 passed**.
- New evaluation tests cover known metrics, zero/missing classes, invalid labels,
  incompatible dataset comparisons, saved results, and mid-run model changes.
- Ruff lint/format checks and whitespace checks passed.
- Live hosted LLM tests remain off without credentials. Heavy benchmark pytest remains
  opt-in; prior performance benchmark reports are separate from F1 evaluation.

See README for local/deployed CLI commands and `.github/workflows/evaluate.yml`
for the prepared repeatable evaluation job. The visible frontend metrics now identify
macro-F1 and this measured dataset instead of presenting the inherited AUC claim.
