"""Serialize model use, including when the HTTP task is cancelled."""

import asyncio
import time
from collections.abc import Callable
from typing import Any


async def run_model_task(
    state: Any, function: Callable, *args: Any, timings: dict | None = None
) -> Any:
    queued = time.perf_counter()
    async with state.inference_lock:
        if timings is not None:
            timings["queue"] = (time.perf_counter() - queued) * 1000
        worker = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Cancelling an await does not stop the thread. Keep the model locked
            # until the existing operation finishes, then propagate cancellation.
            await worker
            raise
