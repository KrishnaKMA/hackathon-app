# NeuroDetect V2 implementation and verification report

Verified locally on October 26, 2026 (America/Toronto).
Repository baseline: `0468514` from `KrishnaKMA/hackathon-app`; changes are local and uncommitted.
No production deployment or remote GitHub Actions run is claimed.

## Architecture and implementation

The existing app was upgraded in place. Original `.keras` model files, class order
(`glioma`, `meningioma`, `notumor`, `pituitary`), RGB conversion, 224x224 resize and
[-1,1] normalization are preserved. The static frontend keeps its existing design.

- `app.py` owns FastAPI, CORS, lifespan, background loading and centralized generic errors.
- `model_loader.py` imports TensorFlow in a worker, loads weights once, creates a fixed-shape
  compiled inference function, and warms it three times before publishing readiness.
- Routers handle HTTP concerns and schemas; ML, Grad-CAM, LLM and serialized worker logic
  live in independently testable services. Legacy endpoint aliases remain.
- CPU work uses threads and one model lock. Cancellation does not release that lock while
  its background model operation is still running. Health/chat remain independent.
- Model predictions preserve all response fields and use cautious model-prediction wording.
- True Grad-CAM replaces the former input saliency algorithm. It targets MobileNetV2's
  final convolution (`Conv_1`), reuses the trained head, computes pre-softmax gradients,
  and returns a dimension-preserving PNG data URL. It is not a tumor-location claim.
- Groq is isolated in `llm_service.py`, defaulting to `openai/gpt-oss-20b`. Async requests
  have 5-second connection and 30-second response timeouts. Missing keys, timeouts,
  rate limits, malformed responses and unsupported providers degrade only chat.
- Classification context is allowlisted to known classes and finite scores. Images and
  arbitrary context/identity fields are never sent to the LLM. User questions are text
  sent to Groq; the UI/docs direct users to avoid confidential identifiers.
- The frontend renders classification first and then requests heatmap and explanation
  independently. Each has retry/error states. Upload generation IDs and abort signals
  prevent stale results, while duplicate in-flight prediction clicks are disabled.
- Actual image contents, upload bytes and decoded pixel counts are validated. CORS uses
  configured origins. Pydantic schemas constrain outputs and questions. Internal failures
  return generic errors. Error text is rendered as text rather than injected HTML.

## Environment and methodology

Windows 11; Intel Core Ultra 7 255U, 14 logical CPUs; Python 3.13.14; TensorFlow 2.21.0; Keras 3.15.1. CPU execution, not GPU.

Inference comparison runs use fresh processes for each threading configuration,
three warm-ups and 30 timed, single-image repetitions with `perf_counter()`. The same
first supplied WebP is preprocessed before inference-only timing. Agreement uses all
11 supplied images. Directory-label evaluation uses the first ten sorted JPEGs per
existing class directory (40 total), with labels unchanged. Powersaving, browser load,
and thermal variation were not experimentally controlled; comparisons are local evidence.

Full endpoint measurements upload the first fixture to the running API, discard three
warm-ups, and time 30 HTTP requests. Server-Timing separates route stages. Route total
starts after FastAPI has parsed the multipart request and ends before framework JSON
serialization; HTTP roundtrip includes both framework and loopback network overhead.
Response preparation times are not complete JSON serialization measurements. Upload
read and Pillow validation are combined; decoding/resizing/normalization are combined
as preprocessing. Separating those substeps further was not instrumented.

Browser durations are read from the visible prediction status in the real in-app
browser. Each includes fetch and JSON parsing; optional stages finish before the next
measurement. Ten runs are measured; this is a small latency sample, not a load test.

## Inference comparison

| Threads (intra/inter) | Implementation | Median ms | P95 ms | n | User-case class agreement |
|---|---|---:|---:|---:|---:|
| default/default | original_predict | 141.67 | 177.07 | 30 | 100.0% |
| default/default | direct_keras | 258.08 | 316.99 | 30 | 100.0% |
| 1/1 | original_predict | 177.86 | 208.94 | 30 | 100.0% |
| 1/1 | direct_keras | 296.54 | 329.76 | 30 | 100.0% |
| 2/1 | original_predict | 81.01 | 109.27 | 30 | 100.0% |
| 2/1 | direct_keras | 139.65 | 286.66 | 30 | 100.0% |
| 4/1 | original_predict | 179.23 | 262.30 | 30 | 100.0% |
| 4/1 | direct_keras | 269.76 | 319.24 | 30 | 100.0% |
| 4/1 | compiled_keras | 21.42 | 23.77 | 30 | 100.0% |
| 4/1 | tflite_float32 | 5.82 | 6.94 | 30 | 100.0% |
| 4/1 | tflite_dynamic_range | 40.88 | 90.08 | 30 | 81.8% |

The eager direct call was slower than `model.predict` here. Compiled Keras is the
selected default because it is fast while retaining the same model for gradients.
The four-thread compiled outputs matched all 11 baseline class predictions with zero
maximum probability difference in that run. Other thread settings had at most a few
millionths probability difference between eager and graph execution.

The local 150 ms warmed inference target is met. This does not establish a 150 ms
Render target: deployment hardware was not accessible for benchmarks. The user's
reported 350 ms was not treated as a fresh baseline; the measured default-thread
`model.predict` median was 141.67 ms.

Default-thread model-load measurement: 2010.8 ms; first `model.predict`: 1833.3 ms. The final API logged about 8.03 seconds for TensorFlow import, model load, compilation and warm-up combined. Hosting spin-up time is additional.

### TensorFlow Lite decision

Both float32 and dynamic-range quantized variants were actually converted and run.
Float32: 10,330,472 bytes, all 11 classes agree with Keras, max score difference about
0.0000135, labelled sample accuracy 39/40. Dynamic range: 2,895,912 bytes, only 9/11
classes agree and labelled accuracy 38/40, so this quantization is rejected. No full
integer-quantized variant was adopted without representative calibration/validation.

TFLite float32 is promising but is not enabled in the API. It does not replace the
Keras gradient model needed for Grad-CAM, and compiled Keras already meets the local
target. The mixed conversion/evaluation process RSS was about 663 MiB during float32
measurements versus about 419 MiB for compiled Keras in that process. These are RSS
snapshots containing conversion allocations, NOT isolated TFLite runtime peaks.
Per-variant isolated peak memory remains unmeasured; the raw process high-water mark
is recorded in `benchmark-threads-4.json`.

The final single-model serving process after the MRI/Grad-CAM workload used 450.5 MiB RSS and reached 543.2 MiB peak working set on this workstation. A 512 MiB allocation has insufficient margin for this observed Windows workload. Linux/Render memory must be measured before choosing a plan; no paid plan was created.

## Complete request timings

| Metric | Median ms | P95 ms | n |
|---|---:|---:|---:|
| http_roundtrip | 63.22 | 75.49 | 30 |
| upload_validate | 1.10 | 1.54 | 30 |
| queue | 0.01 | 0.01 | 30 |
| preprocess | 19.39 | 24.47 | 30 |
| inference | 37.60 | 48.89 | 30 |
| response | 0.07 | 0.10 | 30 |
| total | 58.70 | 70.48 | 30 |
| heatmap_http_roundtrip | 895.96 | 1098.44 | 11 |
| frontend observed prediction | 73.00 | 87.10 | 10 |

The server's inference timings are higher than the isolated microbenchmark because
thread execution, CPU/browser activity and process conditions differ. Preprocessing
was preserved for compatibility instead of changing interpolation or normalization.
Heatmaps remain much more expensive than classification, so they do not gate its display.
Raw endpoint and browser samples are stored alongside this report.

## Existing test material and classification outcomes

All 11 supplied WebP images are copied byte-for-byte from `Downloads/code` with
SHA-256 hashes in `tests/fixtures/user_cases/manifest.json`. Files 1/2 and 7/8 are
identical pairs; there are nine unique images. No user-provided clinical labels were
present. Expected clinical classes remain null instead of being inferred from appearance.

At the user's request, the deployed frontend's API configuration was inspected and
its backend was called with each supplied image. The first attempt timed out during
cold start. After liveness returned, all 11 captures succeeded. Capture timestamp,
backend URL and exact responses are preserved in `deployed_predictions.json`. These
are deployed-model regression references, not independent medical ground truth.

| File | Deployed predicted class | Confidence % | Local class agreement | Heatmap |
|---|---|---:|---|---|
| 1.webp | notumor | 86.41 | yes | valid PNG, original dimensions |
| 2.webp | notumor | 86.41 | yes | valid PNG, original dimensions |
| 3.webp | meningioma | 49.03 | yes | valid PNG, original dimensions |
| 4.webp | notumor | 100.00 | yes | valid PNG, original dimensions |
| 5.webp | notumor | 99.97 | yes | valid PNG, original dimensions |
| 6.webp | meningioma | 82.19 | yes | valid PNG, original dimensions |
| 7.webp | meningioma | 61.46 | yes | valid PNG, original dimensions |
| 8.webp | meningioma | 61.46 | yes | valid PNG, original dimensions |
| 9.webp | meningioma | 82.47 | yes | valid PNG, original dimensions |
| 10.webp | pituitary | 48.21 | yes | valid PNG, original dimensions |
| 11.webp | notumor | 56.43 | yes | valid PNG, original dimensions |

Separate labelled regression accuracy: **39/40 (97.5%)**. This small convenience sample cannot establish clinical accuracy. The unchanged label comparison exposed:

- `test_images\kaggle_samples\meningioma\Te-me_0019.jpg`: directory label `meningioma`, model predicts `notumor`.

The regression check preserves a minimum of the measured 39/40 baseline; it does not
change a label to force the model to be correct. Clinical accuracy for the 11 user
images is unscored because independent labels were not supplied.

## Executed tests

- Lightweight pytest: 25 passed, 32 intentionally skipped, 0 failed/errors, 57 collected.
- Pytest with real TensorFlow integration: **50 passed, 7 skipped, 0 failed/errors**, 57 collected. These overlap the lightweight tests; do not add their totals.
- Measured backend line coverage for the integration invocation: 96.7%.
- Frontend Node/DOM tests: **3 passed, 0 failed**. They test progressive results,
  duplicate-click suppression, stale classification/optional replies, and independent failures.
- Ruff lint and format checks passed. `git diff --check` passed.
- Real running API checks covered 11 predictions and 11 Grad-CAM endpoint uploads,
  valid PNG encoding and source dimensions. The real browser showed classification
  before optional completion and retained it while live AI lacked credentials.
- Six live LLM tests are intentionally skipped because no Groq credential is configured.
  One noisy performance pytest test is intentionally skipped; separate benchmark scripts
  WERE executed, including real TFLite conversion/evaluation and HTTP/browser timing.
- Mocked LLM tests validate request structure, context filtering, response contracts,
  malformed/empty results, missing keys, timeouts and rate limits. They cannot validate
  the factual quality of unexecuted live answers. No live LLM success is claimed.
- Third-party deprecation warnings remain from Starlette/httpx and TensorFlow/gast on
  Python 3.13; they did not cause test failures.

## Deployment and how to run

The Render blueprint uses the repository-root entry point, `$PORT`, one Uvicorn worker,
and `/api/ready` as its HTTP health check. Configure `GROQ_API_KEY` privately,
`GROQ_MODEL=openai/gpt-oss-20b`, and `ALLOWED_ORIGINS` to the exact deployed frontend.
Set the separate frontend backend URL once in `frontend/js/config.js`. The original
model remains bundled. Local 4/1 CPU threading is configurable and must be retuned
on Render. The prepared blueprint has not been deployed.

GitHub Actions is configured for Ruff, lightweight pytest, frontend tests, JUnit and
coverage artifacts. Heavy model integration is an explicit workflow-dispatch option.
No production key is required. Remote execution will occur only after the user pushes
these local changes; no remote CI result is claimed.

Install/run/test commands and all environment variables are in the root README.
A basic Windows local run is:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --env-file .env
.\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory frontend
.\.venv\Scripts\python.exe -m pytest --integration
npm test --prefix frontend
```

Known limits: no independent labels for the 11 user cases; no clinical validation;
one misclassified directory-labelled sample; no live Groq verification without a key;
no deployed hardware performance/memory test; no remote CI run/deploy; no isolated
per-variant TFLite peak measurement; finite local samples are not concurrency/load
capacity guarantees. Large gradient workloads still require memory headroom.

## References consulted

- [Keras Grad-CAM implementation](https://keras.io/examples/vision/grad_cam/)
- [TensorFlow Lite conversion implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/lite.py)
- [Groq model catalogue](https://console.groq.com/docs/models)
- [Groq free-plan limits](https://console.groq.com/docs/rate-limits)
- [Render readiness health checks](https://render.com/docs/health-checks)
- [Render Blueprint reference](https://render.com/docs/blueprint-spec)
- [Render free-service cold starts](https://render.com/docs/free)

## Files created or updated

Full repository-relative manifest (including the prior modularization in this chat)
is in `files-changed.json`. No trained weight file was changed. Generated PNGs, TFLite
models, log files, virtual environments and node_modules are intentionally ignored.

- `.gitignore` (updated)
- `backend/app.py` (updated)
- `backend/config.py` (updated)
- `backend/model_loader.py` (updated)
- `backend/requirements.txt` (updated)
- `backend/utils.py` (updated)
- `frontend/index.html` (updated)
- `frontend/js/api.js` (updated)
- `frontend/js/chat.js` (updated)
- `frontend/js/pages/detector.js` (updated)
- `frontend/js/ui.js` (updated)
- `frontend/partials/footer.html` (updated)
- `frontend/partials/home.html` (updated)
- `requirements.txt` (updated)
- `.env.example` (created)
- `.github/workflows/tests.yml` (created)
- `README.md` (created)
- `backend/routes/__init__.py` (created)
- `backend/routes/chat.py` (created)
- `backend/routes/health.py` (created)
- `backend/routes/heatmap.py` (created)
- `backend/routes/predict.py` (created)
- `backend/routes/uploads.py` (created)
- `backend/schemas/__init__.py` (created)
- `backend/schemas/chat.py` (created)
- `backend/schemas/prediction.py` (created)
- `backend/services/__init__.py` (created)
- `backend/services/chat_service.py` (created)
- `backend/services/heatmap_service.py` (created)
- `backend/services/llm_service.py` (created)
- `backend/services/ml_service.py` (created)
- `backend/services/runtime.py` (created)
- `frontend/js/config.js` (created)
- `frontend/package-lock.json` (created)
- `frontend/package.json` (created)
- `frontend/tests/detector.test.cjs` (created)
- `pytest.ini` (created)
- `render.yaml` (created)
- `reports/UPGRADE_REPORT.md` (created)
- `reports/api-final.xml` (created)
- `reports/benchmark-threads-0.json` (created)
- `reports/benchmark-threads-1.json` (created)
- `reports/benchmark-threads-2.json` (created)
- `reports/benchmark-threads-4.json` (created)
- `reports/browser-timing.json` (created)
- `reports/coverage.xml` (created)
- `reports/deployed-reference.json` (created)
- `reports/endpoint-benchmark-initial.json` (created)
- `reports/endpoint-benchmark.json` (created)
- `reports/environment.json` (created)
- `reports/files-changed.json` (created)
- `reports/integration.xml` (created)
- `reports/labelled-regression.json` (created)
- `reports/unit.xml` (created)
- `requirements-test.txt` (created)
- `ruff.toml` (created)
- `scripts/benchmark.py` (created)
- `scripts/benchmark_endpoints.py` (created)
- `scripts/capture_deployed.py` (created)
- `tests/conftest.py` (created)
- `tests/fixtures/user_cases/1.webp` (created)
- `tests/fixtures/user_cases/10.webp` (created)
- `tests/fixtures/user_cases/11.webp` (created)
- `tests/fixtures/user_cases/2.webp` (created)
- `tests/fixtures/user_cases/3.webp` (created)
- `tests/fixtures/user_cases/4.webp` (created)
- `tests/fixtures/user_cases/5.webp` (created)
- `tests/fixtures/user_cases/6.webp` (created)
- `tests/fixtures/user_cases/7.webp` (created)
- `tests/fixtures/user_cases/8.webp` (created)
- `tests/fixtures/user_cases/9.webp` (created)
- `tests/fixtures/user_cases/deployed_predictions.json` (created)
- `tests/fixtures/user_cases/manifest.json` (created)
- `tests/test_api.py` (created)
- `tests/test_heatmap.py` (created)
- `tests/test_llm.py` (created)
- `tests/test_model_loading.py` (created)
- `tests/test_performance.py` (created)
- `tests/test_prediction.py` (created)
