import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../Server"))
)

MOCK_YOLO_INSTANCE_FOR_TESTS = MagicMock()
MOCK_YOLO_INSTANCE_FOR_TESTS.names = {0: "car", 1: "person"}


@pytest.fixture(scope="session", autouse=True)
def mock_get_yolo_model_globally(request):
    patcher = patch(
        "neopark_server.get_yolo_model", return_value=MOCK_YOLO_INSTANCE_FOR_TESTS
    )

    patcher.start()

    request.config.MOCK_YOLO_FOR_TESTS = MOCK_YOLO_INSTANCE_FOR_TESTS

    yield

    patcher.stop()


_flask_app_global_cache = None


def get_flask_app_instance_val():
    global _flask_app_global_cache
    if _flask_app_global_cache is None:
        from neopark_server import app as actual_flask_app

        _flask_app_global_cache = actual_flask_app
        _flask_app_global_cache.mock_yolo_instance = MOCK_YOLO_INSTANCE_FOR_TESTS
    return _flask_app_global_cache


@pytest.fixture(scope="session")
def app_instance(mock_get_yolo_model_globally):
    flask_app = get_flask_app_instance_val()
    flask_app.config.update(
        {
            "TESTING": True,
            "PROMETHEUS_ENABLE_MIDDLEWARE": False,
        }
    )
    yield flask_app


@pytest.fixture
def client(app_instance):
    with app_instance.test_client() as test_client:
        yield test_client


@pytest.fixture
def runner(app_instance):
    return app_instance.test_cli_runner()


@pytest.fixture(autouse=True)
def mock_prometheus_metrics_classes(app_instance, monkeypatch):
    mock_gauge_instance = MagicMock()
    monkeypatch.setattr(
        "neopark_server.Gauge", MagicMock(return_value=mock_gauge_instance)
    )
    mock_histogram_instance = MagicMock()
    monkeypatch.setattr(
        "neopark_server.Histogram", MagicMock(return_value=mock_histogram_instance)
    )
    mock_counter_instance = MagicMock()
    monkeypatch.setattr(
        "neopark_server.Counter", MagicMock(return_value=mock_counter_instance)
    )
    app_instance.prom_gauge_mock = mock_gauge_instance
    app_instance.prom_histogram_mock = mock_histogram_instance
    app_instance.prom_counter_mock = mock_counter_instance


@pytest.fixture
def clean_areas_data_fixture(app_instance):
    from neopark_server import areas_data

    for area_id_key in list(areas_data.keys()):
        areas_data[area_id_key]["latest_detection"] = {}
        areas_data[area_id_key]["latest_frame"] = None
        areas_data[area_id_key]["processed_frame"] = None
        areas_data[area_id_key]["last_frame_time"] = None
        areas_data[area_id_key]["connection_status"] = False
    yield areas_data
