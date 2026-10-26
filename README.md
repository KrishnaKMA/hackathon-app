> Free-hosting update: Render now uses the validated float32 LiteRT export, without
> TensorFlow in the serving process. Original Keras weights and reference backend
> remain available. See the free-runtime section below; older performance reports
> describe the earlier Keras server.

# NeuroDetect V2

An educational MRI classification prototype using the original MobileNetV2 weights,
FastAPI, and the existing HTML/CSS/vanilla JavaScript interface. This is not a
clinically validated diagnostic system. Model confidence is not guaranteed accuracy;
a qualified professional must interpret actual MRI scans.

## Local installation and launch

Python 3.12 or 3.13, 64-bit, is recommended. This workstation was tested with
Python 3.13.14, TensorFlow 2.21.0, Keras 3.15.1, and Windows 11. Native Windows
TensorFlow uses the CPU. No npm build is required to serve the frontend.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
Copy-Item .env.example .env  # only for first setup; preserve existing secrets
# Edit .env to add GROQ_API_KEY, then:
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --env-file .env
```

In a second terminal:

```powershell
.\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

Open http://127.0.0.1:5500 and http://127.0.0.1:8000/docs. On macOS/Linux use
`.venv/bin/python` instead. Restart the API after changing environment variables.
Do not run multiple API instances on port 8000. Existing local setup logs, when
started with redirected output, are `api.log`, `api-error.log`, and `frontend*.log`.

## Architecture

```text
backend/
  app.py                 FastAPI setup, lifespan, background initialization
  config.py              Environment settings and unchanged class metadata
  model_loader.py        One Keras model per process; compile and warm before ready
  utils.py               Original RGB/224x224/[-1,1] preprocessing
  schemas/               Typed prediction, heatmap, and chat responses
  routes/                HTTP validation, readiness, endpoint orchestration
  services/
    ml_service.py        Classification and original probability response fields
    heatmap_service.py   True convolution-layer Grad-CAM
    llm_service.py       Async Groq provider and allowlisted prediction context
    runtime.py           Serialized CPU work; cancellation-safe model access
    chat_service.py      Compatibility import for the former module name
frontend/
  js/config.js           Optional separate backend URL
  js/api.js              One API client; canonical /api endpoints and request timing
  js/pages/detector.js   Independent prediction, Grad-CAM, explanation states
  tests/                 DOM-level progressive rendering/regression tests
scripts/                 Repeatable isolated CPU/TFLite and endpoint benchmarks
reports/                 Measured benchmark JSON, test XML, and upgrade report
```

`app.py` starts a managed background task; TensorFlow import, model loading,
compilation and three warm-up runs happen in a worker thread. `app.state.model`
is published only after warm-up succeeds. States are `loading`, `ready`, `error`.
Shutdown waits for the initialization thread to finish; Python cannot safely kill it.
A failed initialization keeps liveness available, reports a generic error, and leaves
prediction/readiness unavailable. Detailed failures are for backend logs only.

Prediction and Grad-CAM share one lock. Worker threads do not block health or chat.
If an awaiting request is cancelled, its worker finishes before the model lock is
released. Each process owns one model; deployment uses one worker to limit memory.

## API

| Method | Canonical URL | Response |
|---|---|---|
| GET | `/api/health` | Liveness, model status, initialization duration, class metadata |
| GET | `/api/ready` | 200 after warm-up; otherwise 503 |
| POST | `/api/predict` | Multipart `file`; class, confidence and four probabilities |
| POST | `/api/heatmap` | Multipart `file`; `{ "heatmap": "data:image/png;base64,..." }` |
| POST | `/api/chat` | JSON `message`, optional `context`; `response/status/error` |

Original `/health`, `/ready`, `/predict`, `/heatmap`, `/chat` remain compatibility
aliases. `/` returns basic liveness. Predict responses include `Server-Timing` for
upload/validation, preprocessing, queueing, inference, response preparation and route
total. The browser displays its own full prediction request duration.

JPEG, PNG, WebP and BMP are validated by actual contents. Default limits are 10 MiB
and 4 million decoded pixels. Invalid/empty images return 400, unsupported formats
415, oversized uploads 413, malformed requests 422, and unready models 503. Unexpected
errors return a generic 500. A hosted chat failure returns a degraded chat response,
independently of the model.

## Environment variables

| Variable | Default / use |
|---|---|
| `LLM_PROVIDER` | `groq`; provider dispatch is isolated; unsupported providers degrade explicitly |
| `GROQ_API_KEY` | Required for live chat; backend only |
| `GROQ_MODEL` | `openai/gpt-oss-20b`; old `CHAT_MODEL` remains a fallback |
| `MODEL_BACKEND` | `keras` locally; `tflite` for the validated free Render runtime |
| `MODEL_PATH` | Bundled `backend/models/brain_tumor_multiclass.keras` when blank |
| `INFERENCE_MODE` | `compiled`; alternatives `predict` and `direct` for comparison |
| `TF_INTRA_THREADS` | `4`, based on local benchmarks; `0` lets TensorFlow choose |
| `TF_INTER_THREADS` | `1`; set before TensorFlow initializes |
| `ALLOWED_ORIGINS` | Local frontend on localhost/127.0.0.1 port 5500; comma-separated |
| `CORS_ORIGINS` | Legacy fallback when `ALLOWED_ORIGINS` is absent |
| `MAX_UPLOAD_MB` | `10` |
| `MAX_IMAGE_PIXELS` | `4000000` |
| `LOG_LEVEL` | `INFO` |

Get a Groq key at https://console.groq.com/keys. Its GPT-OSS models have a
rate-limited free plan: https://console.groq.com/docs/rate-limits. Limits and model
availability can change. No key is included; no live answer is fabricated if missing.
The old default Gemini call has been replaced, not run alongside the new provider.

Only allowlisted class names and finite probability scores from context go to the
LLM. Extra context fields, names, IDs, arbitrary instructions and image contents
are dropped. User questions are sent as text to Groq: use general questions and do
not enter personal identifiers or confidential medical information. The educational
prompt distinguishes confidence from accuracy, prohibits diagnosis/treatment decisions,
and never assumes the LLM saw the image. Prompts reduce risk but cannot guarantee
medical correctness. AI output still needs appropriate human judgment.

## Frontend behavior and configuration

Classification renders before optional Grad-CAM and explanation requests start.
Each optional stage has its own loading/error/retry state; a failure keeps prediction
results visible. A new upload clears old context, aborts old HTTP requests and changes
a request generation ID. Stale responses cannot overwrite the new image. Prediction
buttons prevent duplicate in-flight clicks, and heatmap toggling reuses the returned
image rather than requesting another prediction.

The frontend checks readiness once on opening. During cold starts it shows a loading
message; users can retry a prediction. There is no artificial keep-alive polling.

Localhost defaults to http://127.0.0.1:8000; hosted pages use the existing Render API configured centrally in `js/config.js`.
For a different frontend/backend deployment, edit `frontend/js/config.js`:

```javascript
window.NEURODETECT_API_BASE_URL = "https://your-api.onrender.com";
```

This is the sole deployment override. Do not put secrets here. The frontend remains
static and can be served independently. After editing partial HTML, regenerate the
index with `python frontend/scripts/build_index.py`; script configuration is included
in the footer partial as well as the current generated index.

## Grad-CAM

V1's implementation was input-gradient saliency. V2 uses the last `Conv2D` layer
(`Conv_1`) inside the existing nested `mobilenetv2_1.00_224` backbone. A cached
extractor returns convolutional activations and backbone output; the original head
is reused to compute pre-softmax class scores without mutating model weights or
activation functions. Mean gradients weight feature channels; a ReLU and safe
normalization produce the map. The overlay preserves the source dimensions and
existing data-URL response format. A zero positive map is allowed and shown without
added color. This is model influence, not proof of tumor boundaries or location.
Reference: https://keras.io/examples/vision/grad_cam/.

## Performance choices

Measured locally, not on Render: default `model.predict` median 141.7 ms;
four-thread compiled Keras median 21.4 ms. Complete local API requests measured
63.2 ms median / 75.5 ms P95 over 30 runs; browser request durations ranged 57-88 ms
in ten runs. See `reports/UPGRADE_REPORT.md` for all configurations and caveats.
The reported earlier 350 ms was not used as a measured baseline.

Eager direct calls were slower here, so they were not chosen. `tf.function` keeps
Keras weights and gradients available with a compiled prediction graph. TFLite
float32 measured 5.8 ms but would require a second runtime/model for Grad-CAM;
compiled Keras already meets the local target. Dynamic-range quantization changed
2/11 user-case classes and was rejected. Conversion-process memory is reported
separately from isolated serving memory. No 150 ms guarantee is made for Render.

RGB conversion, resize, channel order and normalization remain unchanged. Upload
validation necessarily inspects the image before preprocessing; the model is never
reloaded for requests. Heatmaps are a separate, more expensive operation.

## Tests and the 11 supplied cases

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest --junitxml=reports/unit.xml
.\.venv\Scripts\python.exe -m pytest --integration --junitxml=reports/integration.xml
.\.venv\Scripts\ruff.exe check backend tests scripts
.\.venv\Scripts\ruff.exe format --check backend tests scripts
npm ci --prefix frontend
npm test --prefix frontend
```

Ordinary pytest runs need no TensorFlow import or production API key. Integration
runs require the runtime dependencies and bundled weights. Live chat tests are
explicit: export `GROQ_API_KEY` in the shell, then `pytest --live tests/test_llm.py`.
Pytest itself does not load `.env`; API startup uses Uvicorn's `--env-file` option.

The exact 11 WebP files from `Downloads/code` are preserved in
`tests/fixtures/user_cases`. `manifest.json` records SHA-256 and has null clinical
labels because no labels were supplied. Files 1/2 and 7/8 are duplicate pairs.
`deployed_predictions.json` records outputs obtained from the user-specified hosted
frontend's backend, including provenance and capture time. These are regression
references, not medical ground truth. Tests compare all 11 local outputs to them,
check preprocessing/probabilities, and validate 11 real Grad-CAM maps and PNGs.

A separate test evaluates the fixed 40-image manifest (ten unchanged labels per class). Baseline was 39/40; the optimized runtime must preserve at
least that baseline on this small sample. Labels are unchanged. These small samples
cannot establish clinical accuracy. Existing UI claims about historical 89.7% accuracy
are inherited training claims, not a new result established by this upgrade.

Reproduce benchmarks explicitly (keep other CPU work quiet):

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --threads 0 --runs 30
.\.venv\Scripts\python.exe scripts/benchmark.py --threads 1 --runs 30
.\.venv\Scripts\python.exe scripts/benchmark.py --threads 2 --runs 30
.\.venv\Scripts\python.exe scripts/benchmark.py --threads 4 --runs 30 --lite
# Requires the API running and ready:
.\.venv\Scripts\python.exe scripts/benchmark_endpoints.py
# Explicitly regenerate deployed regression references after reviewing upstream changes:
.\.venv\Scripts\python.exe scripts/capture_deployed.py
```

Benchmarks write JSON and report median, P95 and sample count. The performance
pytest marker is off by default (`pytest --performance` runs a CPU benchmark).
Timing is not a CI pass/fail threshold. TFLite files/logs/screenshots are local
artifacts and ignored by Git. Original Keras weights remain untouched.

## CI and Render

`.github/workflows/tests.yml` runs Ruff, lightweight Python tests, frontend DOM tests,
coverage and JUnit output on pushes/PRs. TensorFlow regression tests and the Render startup smoke test run on pushes/PRs; manual dispatch can optionally skip the heavy job. Normal CI does not require LLM credentials.

`render.yaml` installs backend requirements, runs
`uvicorn backend.app:app --host 0.0.0.0 --port $PORT --workers 1`, and checks
`/api/ready`. Set `GROQ_API_KEY` privately and `ALLOWED_ORIGINS` to the exact hosted
frontend origin (for example `https://frontend-4s5x.onrender.com`). Configure the
frontend's backend URL separately as described above. The bundled model is loaded
from disk; no per-request download is needed. Use hosting with enough memory for
TensorFlow and gradient computation; measure actual peak memory under workload.

Render free services can spin down. Wake-up and model warm-up are distinct from
warmed inference latency. Tune TensorFlow threads on the actual Render hardware;
local results do not prove deployment performance. The blueprint is prepared but
this upgrade does not claim a successful production deployment or remote CI run.

Final local serving memory after Grad-CAM: about 450 MiB resident, 543 MiB peak
working set on Windows. Treat 512 MiB hosting as insufficient margin for this
observed workload and measure the Linux deployment before selecting capacity.


## Full-dataset F1 tracking and deployed evaluation

The 11 unlabelled user examples remain functional/regression checks. They are not
used to estimate F1. `scripts/evaluate.py` evaluates every labelled image in the
four class directories; the default repository test set contains 1,311 images.

```powershell
# Evaluate original weights locally and save an immutable timestamped run:
.\.venv\Scripts\python.exe scripts/evaluate.py --model-version v2
# Compare a new model with the saved baseline, using exactly the same dataset:
.\.venv\Scripts\python.exe scripts/evaluate.py --compare reports/evaluations/baseline.json
# Black-box evaluation of an updated, deployed API:
.\.venv\Scripts\python.exe scripts/evaluate.py --api-url https://your-api.onrender.com --model-version deployment-commit --compare reports/evaluations/baseline.json --output reports/deployed-evaluations
```

Each run stores per-class precision, recall, F1 and support; macro-F1, weighted-F1,
accuracy and a confusion matrix (rows=true, columns=predicted). Undefined divisions
use zero and missing classes are explicitly flagged. JSON includes individual
predictions, dataset file hashes, dataset fingerprint, model SHA-256, source commit,
working-tree status, run time and version. Markdown summaries accompany each JSON.
Comparisons reject a changed dataset or class order instead of presenting misleading
deltas. Deployed runs require `/api/ready` to expose a model hash and reject runs if
that hash changes mid-evaluation. Failed/incomplete runs do not replace `latest.json`.

Full local evaluation on October 26, 2026 measured macro-F1 **0.8837**, weighted-F1
**0.8890**, and accuracy **89.09%**. A second black-box run against the local serving
API evaluated all 1,311 files again and produced zero metric deltas against baseline.
See `reports/evaluations/latest.md`, `reports/serving-evaluations/latest.md` and
`reports/EVALUATION_TRACKING.md`. Labels come from existing directories; independent
clinical validity, population representativeness and absence of training overlap
still require a dataset audit. These are not clinical performance claims.

`.github/workflows/evaluate.yml` is a repeatable deployed-API evaluation job with
three triggers: successful GitHub deployment status, manual dispatch, and weekly
Monday evaluation. It uses no TensorFlow or LLM credentials, compares the fixed
baseline and uploads versioned JSON/Markdown artifacts for 90 days. Metrics are
published in the Actions run summary. Configure repository variable
`EVALUATION_API_URL` with the backend base URL, push the workflow to the default
branch, and ensure the updated API is deployed. A Render deployment must emit a
GitHub deployment status (through an existing deployment integration or webhook
bridge) for the immediate post-deployment trigger; the workflow does not create
that integration. Without it, manual/weekly runs provide repeatable evaluation but
are not evidence of evaluation after every deployment. Monitoring is prepared,
not claimed active from these local changes. Scheduled runs are substantive dataset
evaluations, not keep-alive requests. Dataset-specific baselines are fixed until a
reviewed model/dataset change; compare arbitrary archived runs with `--compare`.


## Pre-push deployment validation

Runtime dependencies are pinned to versions validated on Windows Python 3.13 and
Linux Python 3.12.10, the version specified in `render.yaml`. The static frontend's
single `js/config.js` setting points hosted pages to the existing Render API while
keeping localhost on port 8000. Render's allowed origins include the existing hosted
frontend. For a custom domain, update this explicit list before deployment.

`python scripts/deployment_smoke.py` launches the exact Render Uvicorn command in a
fresh process with its Blueprint environment defaults. It tests liveness during
loading, readiness after warm-up, model hash, a real prediction and heatmap, rejected
invalid uploads, hosted CORS and chat independence without a key. It runs in the
integration CI job. It does not establish the memory/CPU limits or environment of an
existing Render account. Check its configured root directory, startup command, memory
capacity, CORS values and private Groq key before merging a production deployment.

The existing Azure backend workflow now also targets Python 3.12. The Azure static
frontend explicitly skips app compilation: committed HTML is already built and npm
is used for tests. These changes preserve the original optional deployment workflows.

Final Linux deployment smoke testing observed about **775 MiB peak RSS** with
prediction and Grad-CAM. Do not deploy this measured configuration on a 512 MiB
instance. See `reports/FINAL_VALIDATION.md` for the complete pre-push checks and
remaining Render account verification.


## Free Render runtime (current deployment configuration)

`backend/requirements-render.txt` installs LiteRT instead of TensorFlow. Set
`MODEL_BACKEND=tflite`, `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. The original
MobileNetV2 is exported without quantization to `brain_tumor_float32.tflite`, exposing
both four-class probabilities and the final convolutional activations. The small
`gradcam_head.npz` holds the original BatchNorm and Dense weights. SHA-256 checks bind
these artifacts to the unchanged source Keras file; re-export after changing weights.

Grad-CAM uses the analytic chain rule through BatchNorm, ReLU6, global average pooling
and the three Dense layers. This produces the same convolution-layer Grad-CAM, without
loading TensorFlow for automatic differentiation. Equivalence tests compare probability
arrays and Grad-CAM maps against the Keras reference on the 11 supplied images and
representatives of all four classes. Full evaluation preserved all 1,311 predicted
classes and F1 metrics. The original Keras backend remains the local default.

Fresh Linux serving without any TensorFlow package measured about 154 MiB peak RSS for
normal prediction/heatmap checks and about 237 MiB after three maximum-sized (4 million
pixel) heatmap requests. In-place overlay blending bounds image memory. These replace
the previous 775 MiB Keras requirement for the configured free deployment; actual Render
usage and timing must still be checked after deployment.

The same free service also serves the separate frontend files at `/ui/`. Its API root
and canonical `/api/...` routes remain intact. The verified account backend is
`https://hackathon-app-adb9.onrender.com`; the older `88w3` URL is no longer the configured
production target. No paid compute change is required.

Re-export in a full development environment with `python scripts/export_lite.py`.
Evaluate with `MODEL_BACKEND=tflite python scripts/evaluate.py --compare reports/evaluations/baseline.json`.
Measure the deployed API using `python scripts/benchmark_endpoints.py --api-url URL
--runs 30 --output reports/render-benchmark.json --expected-model-sha SOURCE_SHA256`.
