import io
import threading
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.services import llm_service as chat_service


class FakeModel:
    def __call__(self, x, training=False):
        return self.predict(x)

    def predict(self, x, verbose=0):
        assert x.shape == (1, 224, 224, 3)
        assert x.min() >= -1 and x.max() <= 1
        return np.array([[0.1, 0.2, 0.6, 0.1]])


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (32, 32)).save(stream, format="PNG")
    return stream.getvalue()


class ApiTests(unittest.TestCase):
    def test_background_loading_and_contract(self):
        release = threading.Event()

        def loader():
            release.wait(10)
            return FakeModel()

        with patch("backend.app.load_model", loader):
            with TestClient(app) as client:
                try:
                    self.assertEqual(
                        client.get("/api/health").json()["model_status"], "loading"
                    )
                    self.assertEqual(client.get("/api/ready").status_code, 503)
                    self.assertEqual(
                        client.post(
                            "/api/predict", files={"file": ("x.png", image_bytes())}
                        ).status_code,
                        503,
                    )
                    app.state.model = FakeModel()
                    self.assertEqual(client.get("/api/ready").status_code, 200)
                    for url in ["/api/predict", "/predict"]:
                        response = client.post(
                            url, files={"file": ("x.png", image_bytes())}
                        )
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json()["predicted_class"], "notumor")
                        self.assertEqual(len(response.json()["all_predictions"]), 4)
                    for content, code in [
                        (b"", 400),
                        (b"not an image", 400),
                        (b"x" * (10 * 1024 * 1024 + 1), 413),
                    ]:
                        self.assertEqual(
                            client.post(
                                "/api/predict", files={"file": ("x.png", content)}
                            ).status_code,
                            code,
                        )
                    with patch(
                        "backend.routes.heatmap.generate_heatmap_image",
                        return_value="data:image/png;base64,test",
                    ):
                        self.assertIn(
                            "heatmap",
                            client.post(
                                "/api/heatmap", files={"file": ("x.png", image_bytes())}
                            ).json(),
                        )
                    self.assertEqual(
                        client.post("/api/chat", json={"message": "   "}).status_code,
                        422,
                    )
                finally:
                    release.set()

    def test_failed_load_stays_live(self):
        with patch("backend.app.load_model", side_effect=RuntimeError("private path")):
            with TestClient(app) as client:
                # Deterministic check of public failure state.
                app.state.model_status = "error"
                self.assertEqual(
                    client.get("/api/health").json()["model_error"],
                    "Model failed to load",
                )
                self.assertEqual(client.get("/api/ready").status_code, 503)


class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key(self):
        with patch.object(chat_service, "CHAT_API_KEY", ""):
            self.assertEqual(
                (await chat_service.answer_question("hello"))["error"],
                "missing_api_key",
            )

    async def test_provider_contract(self):
        real_client = httpx.AsyncClient
        for code, payload, expected in [
            (200, {"choices": [{"message": {"content": "A real answer"}}]}, "ok"),
            (429, {}, "degraded"),
            (200, {}, "degraded"),
            (200, {"choices": [{"message": {"content": ""}}]}, "degraded"),
        ]:

            def handler(request):
                self.assertEqual(request.headers["Authorization"], "Bearer test-key")
                return httpx.Response(code, json=payload)

            transport = httpx.MockTransport(handler)
            with (
                patch.object(chat_service, "CHAT_API_KEY", "test-key"),
                patch.object(
                    chat_service.httpx,
                    "AsyncClient",
                    side_effect=lambda **kw: real_client(transport=transport, **kw),
                ),
            ):
                result = await chat_service.answer_question(
                    "Explain", {"predicted_class": "glioma"}
                )
                self.assertEqual(result["status"], expected)


class ExtendedApiTests(unittest.TestCase):
    def setUp(self):
        self.loader = patch("backend.app.load_model", return_value=FakeModel())
        self.loader.start()
        self.client = TestClient(app, raise_server_exceptions=False)
        self.client.__enter__()
        app.state.model = FakeModel()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.loader.stop()

    def test_cors(self):
        response = self.client.options(
            "/api/predict",
            headers={
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"], "http://127.0.0.1:5500"
        )
        hosted = self.client.options(
            "/api/predict",
            headers={
                "Origin": "https://frontend-4s5x.onrender.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(hosted.status_code, 200)
        rejected = self.client.options(
            "/api/predict",
            headers={
                "Origin": "https://untrusted.invalid",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(rejected.status_code, 400)

    def test_aliases_and_missing_file(self):
        for url in ["/health", "/api/health", "/ready", "/api/ready", "/"]:
            self.assertEqual(self.client.get(url).status_code, 200)
        for url in ["/predict", "/heatmap", "/api/predict", "/api/heatmap"]:
            self.assertEqual(self.client.post(url).status_code, 422)
        for url in ["/chat", "/api/chat"]:
            with patch.object(chat_service, "CHAT_API_KEY", ""):
                self.assertEqual(
                    self.client.post(url, json={"message": "hello"}).json()["error"],
                    "missing_api_key",
                )

    def test_sanitized_internal_error_and_timing(self):
        with patch(
            "backend.routes.predict.predict_scan",
            side_effect=RuntimeError("secret path"),
        ):
            response = self.client.post(
                "/api/predict", files={"file": ("test.png", image_bytes())}
            )
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("secret", response.text)
        response = self.client.post(
            "/api/predict", files={"file": ("test.png", image_bytes())}
        )
        self.assertIn("inference;dur=", response.headers["server-timing"])

    def test_heatmap_errors(self):
        self.assertEqual(
            self.client.post(
                "/api/heatmap", files={"file": ("fake.png", b"garbage")}
            ).status_code,
            400,
        )
        app.state.model = None
        self.assertEqual(
            self.client.post(
                "/api/heatmap", files={"file": ("test.png", image_bytes())}
            ).status_code,
            503,
        )

    def test_actual_format_and_decoded_pixel_limits(self):
        stream = io.BytesIO()
        Image.new("RGB", (8, 8)).save(stream, format="GIF")
        self.assertEqual(
            self.client.post(
                "/api/predict",
                files={"file": ("pretend.jpg", stream.getvalue(), "image/jpeg")},
            ).status_code,
            415,
        )
        with patch("backend.routes.uploads.MAX_IMAGE_PIXELS", 10):
            self.assertEqual(
                self.client.post(
                    "/api/predict", files={"file": ("image.png", image_bytes())}
                ).status_code,
                400,
            )
