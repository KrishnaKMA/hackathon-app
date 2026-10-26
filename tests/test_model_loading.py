import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from backend.app import lifespan, load_model_in_background


def test_ready_only_after_loader_returns():
    async def scenario():
        app = SimpleNamespace(state=SimpleNamespace(model=None, model_status="loading"))
        with patch("backend.app.load_model", return_value=object()):
            await load_model_in_background(app)
        assert app.state.model is not None
        assert app.state.model_status == "ready"
        assert app.state.model_load_seconds >= 0

    asyncio.run(scenario())


def test_failure_keeps_model_unavailable():
    async def scenario():
        app = SimpleNamespace(state=SimpleNamespace(model=None, model_status="loading"))
        with patch("backend.app.load_model", side_effect=RuntimeError("test")):
            await load_model_in_background(app)
        assert app.state.model is None and app.state.model_status == "error"

    asyncio.run(scenario())


def test_lifespan_owns_loading_task():
    async def scenario():
        app = SimpleNamespace(state=SimpleNamespace())
        with patch("backend.app.load_model", return_value=object()):
            async with lifespan(app):
                assert app.state.loading_task.get_name() == "model-initialization"
            assert app.state.loading_task.done()
            assert app.state.model_status == "ready"

    asyncio.run(scenario())


def test_warmup_is_part_of_load():
    import sys
    from unittest.mock import MagicMock

    import numpy as np

    from backend import model_loader

    fake = MagicMock()
    fake.keras.models.load_model.return_value.return_value.numpy.return_value = (
        np.ones((1, 4)) / 4
    )
    fake.keras.models.load_model.return_value.predict.return_value = np.ones((1, 4)) / 4
    with (
        patch.dict(sys.modules, {"tensorflow": fake}),
        patch.object(model_loader, "INFERENCE_MODE", "direct"),
    ):
        loaded = model_loader.load_model()
    assert loaded.call_count == 3


def test_cancelled_request_keeps_inference_serialized():
    import threading

    from backend.services.runtime import run_model_task

    async def scenario():
        state = SimpleNamespace(inference_lock=asyncio.Lock())
        started = threading.Event()
        release = threading.Event()

        def slow():
            started.set()
            release.wait(5)

        first = asyncio.create_task(run_model_task(state, slow))
        await asyncio.to_thread(started.wait, 5)
        first.cancel()
        second = asyncio.create_task(run_model_task(state, lambda: 42))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        try:
            await first
        except asyncio.CancelledError:
            pass
        assert await second == 42

    asyncio.run(scenario())
