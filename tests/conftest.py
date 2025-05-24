# tests/conftest.py
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Tambahkan direktori Server ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Server')))

# Tunda import app sampai sys.path di-set dan mock bisa diaplikasikan
_flask_app_global_cache = None
_mock_yolo_instance_for_tests_cache = None # Variabel global untuk menyimpan mock instance

def get_flask_app_with_mocks():
    global _flask_app_global_cache, _mock_yolo_instance_for_tests_cache
    
    if _flask_app_global_cache is None:
        # Buat instance mock untuk YOLO
        mock_yolo_instance = MagicMock()

        _mock_yolo_instance_for_tests_cache = mock_yolo_instance

        with patch('ultralytics.YOLO', return_value=mock_yolo_instance) as _mock_ultralytics_yolo, \
             patch('neopark_server.YOLO', return_value=mock_yolo_instance, create=True) as _mock_neopark_server_yolo:
            
            # Sekarang aman untuk mengimpor modul aplikasi
            from neopark_server import app as actual_flask_app
            _flask_app_global_cache = actual_flask_app

            _flask_app_global_cache.mock_yolo_instance = _mock_yolo_instance_for_tests_cache
            
    return _flask_app_global_cache


@pytest.fixture(scope='session')
def app_instance():
    """Fixture untuk instance aplikasi Flask (session-scoped) dengan YOLO sudah di-mock."""
    flask_app = get_flask_app_with_mocks()
    flask_app.config.update({
        "TESTING": True,
        "PROMETHEUS_ENABLE_MIDDLEWARE": False, # Jika Anda menggunakan setting ini di app Anda
    })
    yield flask_app # Menyediakan app instance ke tes

@pytest.fixture
def client(app_instance): # Scope default adalah 'function'
    """Fixture untuk Flask test client."""
    with app_instance.test_client() as test_client:
        yield test_client

@pytest.fixture
def runner(app_instance): # Scope default adalah 'function'
    """Fixture untuk Flask CLI runner (jika Anda punya custom CLI commands)."""
    return app_instance.test_cli_runner()

@pytest.fixture(autouse=True) # autouse=True agar fixture ini otomatis dipakai di setiap tes
def mock_prometheus_metrics_classes(app_instance, monkeypatch):
    mock_gauge_instance = MagicMock()
    monkeypatch.setattr('neopark_server.Gauge', MagicMock(return_value=mock_gauge_instance))
    
    mock_histogram_instance = MagicMock()
    # mock_histogram_instance.labels.return_value.observe = MagicMock() # Jika perlu lebih spesifik
    monkeypatch.setattr('neopark_server.Histogram', MagicMock(return_value=mock_histogram_instance))

    mock_counter_instance = MagicMock()
    monkeypatch.setattr('neopark_server.Counter', MagicMock(return_value=mock_counter_instance))
    
    # Lampirkan instance mock metrik ke app_instance agar bisa diakses/diverifikasi dalam tes
    app_instance.prom_gauge_mock = mock_gauge_instance
    app_instance.prom_histogram_mock = mock_histogram_instance
    app_instance.prom_counter_mock = mock_counter_instance


@pytest.fixture # Scope default adalah 'function', cocok untuk state yang direset per tes
def clean_areas_data_fixture(app_instance): # Nama diubah sedikit untuk kejelasan
    from neopark_server import areas_data
        
    # Reset ke state awal yang diketahui untuk setiap area
    for area_id_key in list(areas_data.keys()): # list() untuk menghindari error size change saat iterasi jika ada modifikasi
        areas_data[area_id_key]['latest_detection'] = {}
        areas_data[area_id_key]['latest_frame'] = None
        areas_data[area_id_key]['processed_frame'] = None
        areas_data[area_id_key]['last_frame_time'] = None
        areas_data[area_id_key]['connection_status'] = False
        # Objek 'frame_lock' (threading.Lock) tidak perlu dibuat ulang, biarkan saja.

    yield areas_data # Menyediakan 'areas_data' yang sudah bersih ke fungsi tes