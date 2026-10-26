# Final pre-push validation

Completed locally on October 26, 2026. The production branch is intentionally kept
unchanged while the upgrade is pushed to `neurodetect-v2-validation` for review.

## Executed checks

- Windows Python 3.13.14: 59 passed, 7 intentionally skipped; `final-windows.xml`.
- Linux x86_64 / Python 3.12.10, matching Render: 59 passed, 7 intentionally skipped;
  `final-linux.xml`. Six live LLM tests require a Groq key; one noisy performance
  pytest marker is off because benchmarks were executed separately.
- Clean npm install/audit: zero reported vulnerabilities; 7 frontend tests passed.
- Ruff lint and formatting, pip dependency checks and Git whitespace checks passed.
- Secret-pattern scan found no key/private-key matches in changed text files; `.env`
  remains ignored and is excluded from the push.
- Exact Render Uvicorn command and Blueprint environment were tested in fresh Windows
  and Linux processes. Liveness remained responsive during loading, readiness waited
  for warm-up, and real prediction, Grad-CAM, invalid-upload rejection, missing-key
  chat behavior and hosted CORS all passed.
- A fresh full Linux evaluation scored all 1,311 labelled files and reproduced the
  saved accuracy, macro-F1, weighted-F1 and per-class F1 with zero deltas. The dataset
  fingerprint also matched the Windows baseline.
- Original trained `.keras` files are unchanged.

## Issues found and corrected before pushing

1. Hosted frontend requests could otherwise target its static-site origin. The single
   config now selects the existing Render backend and preserves custom host overrides.
2. The existing hosted frontend is in the explicit CORS defaults and Render Blueprint.
   Frontend URL routing and server CORS have regression coverage.
3. Windows and Linux path sorting selected different 40-image regression samples.
   A fixed manifest preserves the ORIGINAL sample, labels and 39/40 baseline threshold.
   Full-dataset evaluation ordering is explicit and portable for stable fingerprints.
4. Runtime/test dependencies are pinned to the versions validated on both platforms.
5. The older Azure backend workflow now uses Python 3.12; static Azure frontend build
   compilation is skipped because its committed HTML is already generated and npm
   dependencies are for tests. Existing Azure credentials/deployments were not exercised.
6. CI now runs lightweight tests, frontend tests, real TensorFlow integration and the
   deployment startup smoke test on pushes/PRs. Manual dispatch can omit the heavy job.

## Render production boundary

The Render dashboard is not signed in, so its existing root directory, instance size,
startup command, environment variables and automatic deployment settings could not be
read. No production settings were changed and no deployment success is claimed.

**Measured Linux API peak RSS was 775.3 MiB** after prediction and Grad-CAM; resident
memory was about 750.5 MiB. A 512 MiB instance is not sufficient for this measured
configuration. Provision memory above that observed peak with workload headroom and
verify it on the actual Render instance before merging/deploying. No paid service or
capacity change was created. This resource requirement prevents an honest guarantee
that an unknown existing Render plan will work.

Before production merge, verify repository-root build/start paths, one Uvicorn worker,
`/api/ready`, exact deployed frontend origin, adequate memory and private Groq key.
Source checks do not verify external account settings. Keeping main unchanged avoids
triggering a production auto-deployment while these settings are unverified.

Post-deployment F1 evaluation is configured as described in `EVALUATION_TRACKING.md`.
It becomes active only after the workflow is pushed to the default branch and the
API URL/deployment event integration are configured.
