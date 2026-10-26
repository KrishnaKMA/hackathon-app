from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true")
    parser.addoption("--performance", action="store_true")
    parser.addoption("--live", action="store_true")


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker in ("integration", "performance", "live"):
            if marker in item.keywords and not config.getoption("--" + marker):
                item.add_marker(
                    pytest.mark.skip(reason=f"Enable explicitly with --{marker}")
                )


@pytest.fixture(scope="session")
def real_model():
    from backend.model_loader import load_model

    return load_model()


@pytest.fixture(scope="session")
def user_cases():
    import json

    root = Path(__file__).parent / "fixtures/user_cases"
    return root, json.loads((root / "manifest.json").read_text())["cases"]
