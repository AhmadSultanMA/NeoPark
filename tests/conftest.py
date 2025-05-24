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


MOCK_GAUGE_INSTANCE = MagicMock(name="GlobalMockGaugeInstance")
MOCK_HISTOGRAM_INSTANCE = MagicMock(name="GlobalMockHistogramInstance")
MOCK_COUNTER_INSTANCE = MagicMock(name="GlobalMockCounterInstance")


@pytest.fixture(scope="session", autouse=True)
def mock_prometheus_primitives_globally(request):
    patcher_gauge = patch("neopark_server.Gauge", return_value=MOCK_GAUGE_INSTANCE)
    patcher_histogram = patch(
        "neopark_server.Histogram", return_value=MOCK_HISTOGRAM_INSTANCE
    )
    patcher_counter = patch(
        "neopark_server.Counter", return_value=MOCK_COUNTER_INSTANCE
    )

    patcher_gauge.start()
    patcher_histogram.start()
    patcher_counter.start()

    request.config.MOCK_GAUGE_INSTANCE = MOCK_GAUGE_INSTANCE
    request.config.MOCK_HISTOGRAM_INSTANCE = MOCK_HISTOGRAM_INSTANCE
    request.config.MOCK_COUNTER_INSTANCE = MOCK_COUNTER_INSTANCE

    yield

    patcher_gauge.stop()
    patcher_histogram.stop()
    patcher_counter.stop()


@pytest.fixture(autouse=True)
def setup_prometheus_mocks_on_app(app_instance, mock_prometheus_primitives_globally):
    app_instance.prom_gauge_mock = MOCK_GAUGE_INSTANCE
    app_instance.prom_histogram_mock = MOCK_HISTOGRAM_INSTANCE
    app_instance.prom_counter_mock = MOCK_COUNTER_INSTANCE

    MOCK_GAUGE_INSTANCE.reset_mock()
    MOCK_HISTOGRAM_INSTANCE.reset_mock()
    MOCK_COUNTER_INSTANCE.reset_mock()


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
