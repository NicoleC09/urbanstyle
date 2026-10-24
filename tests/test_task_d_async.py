from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from app.tabs.task_d_tab import TaskDTab


def test_generation_runs_off_the_ui_thread_and_is_collected() -> None:
    started = threading.Event()
    release = threading.Event()
    worker_thread: list[int] = []
    tab = TaskDTab.__new__(TaskDTab)
    tab.generator = SimpleNamespace(
        generate=lambda *args, **kwargs: _blocked_generation(
            started, release, worker_thread
        )
    )
    tab.model_error = None
    tab.seed_text = "42"
    tab.seed_value = 42
    tab.seed_focused = False
    tab.class_id = 7
    tab.count_slider = SimpleNamespace(value=3)
    tab.variety_slider = SimpleNamespace(value=0.8)
    tab._generated_surface_cache = {}
    tab.buttons = {
        name: SimpleNamespace(enabled=True)
        for name in ("generate", "random", "poster", "download", "large")
    }
    tab.images = None
    tab.hero_index = 0
    tab.poster_path = None
    tab.message = ""
    tab._executor = ThreadPoolExecutor(max_workers=1)
    tab._future = None
    tab._future_action = None
    tab._followup_action = None

    try:
        ui_thread = threading.get_ident()
        tab.generate()

        assert tab._future is not None
        assert tab.images is None
        assert not tab.buttons["generate"].enabled
        assert started.wait(timeout=2)
        assert worker_thread[0] != ui_thread

        release.set()
        tab._future.result(timeout=2)
        tab.update()

        assert tab.images == ["generated"]
        assert tab.message == "1 muestras generadas · selecciona el héroe"
        assert tab.buttons["generate"].enabled
    finally:
        release.set()
        tab._executor.shutdown(wait=True)


def _blocked_generation(
    started: threading.Event,
    release: threading.Event,
    worker_thread: list[int],
) -> list[str]:
    worker_thread.append(threading.get_ident())
    started.set()
    assert release.wait(timeout=2)
    return ["generated"]