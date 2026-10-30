import httpx
import pytest

from scripts.monitor_production import trigger_deploy, wait_ready


def test_deploy_preserves_secret_query_and_pins_tested_revision():
    def handler(request):
        assert request.url.params["key"] == "test-secret"
        assert request.url.params["ref"] == "a" * 40
        return httpx.Response(200, json={"deploy": {"id": "test"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        trigger_deploy(
            client, "https://api.render.com/deploy/test?key=test-secret", "a" * 40
        )


def test_ready_rejects_previous_revision():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"status": "ready", "model_sha256": "weights", "revision": "old"}
        )
    )
    with httpx.Client(base_url="https://example.test", transport=transport) as client:
        with pytest.raises(RuntimeError, match="expected revision"):
            wait_ready(client, "new", timeout=0)


def test_hook_errors_do_not_disclose_secret():
    transport = httpx.MockTransport(lambda request: httpx.Response(401))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError) as error:
            trigger_deploy(
                client, "https://api.render.com/deploy/test?key=test-secret", "a" * 40
            )
        assert "test-secret" not in str(error.value)
