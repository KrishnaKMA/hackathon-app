# Testing, deployment and monitoring

Pushes and pull requests run Python lint/format checks, frontend tests, Python
unit tests and real-model integration tests, including the free-tier memory smoke
test. Only a push to `main` deploys, and both test jobs must pass first.

Set GitHub repository Actions secret `RENDER_DEPLOY_HOOK_URL` to this Render
service's deploy hook (Settings > Deploy Hook). Never commit or paste the secret
into logs. The pipeline sends the tested full commit SHA as `ref`, so a later push
cannot silently change what this run deploys. Render disables its independent
auto-deploy when a specific commit is deployed through a hook. Keep it disabled
so GitHub's test gate controls releases.

The production job waits up to 20 minutes for `/api/ready` to report the tested
revision via Render's `RENDER_GIT_COMMIT` environment variable. It validates real
prediction responses, heatmap decoding/dimensions, the frontend and invalid-upload
handling. It then evaluates all labelled validation fixtures against the saved
baseline and publishes metrics and reports as 90-day artifacts.

The independent production-monitor workflow runs every six hours or manually.
It allows up to four minutes for a free service to wake, performs the same smoke
checks, and saves three prediction roundtrip and Server-Timing measurements.
These samples are diagnostic observations, not a reliable latency percentile or
an uptime SLA. Scheduled GitHub jobs may be delayed. Checks wake the free service;
they do not keep it continuously awake. Weekly labelled evaluation also remains
enabled through the existing evaluation workflow.

A failed monitor opens one deduplicated GitHub issue; a successful later check
comments and closes that issue. GitHub Actions also records deployment failures;
enable Actions/issue notifications in your GitHub account to receive them. Issues
must be enabled on the repository. This does not install a separate alert service.

A failed post-deployment check marks the workflow failed but does not automatically
roll back: network or provider failures alone should not select an older release.
Use Render's deployment history to roll back to a known-good deployment, then fix
or revert `main` before releasing again. Free-tier limits and external chat-provider
availability still apply. Monitoring does not send chat requests or use Groq keys.
