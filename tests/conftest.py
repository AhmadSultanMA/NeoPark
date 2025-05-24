# tests/conftest.py
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Tambahkan direktori Server ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Server')))

# Tunda import app sampai sys.path di-set dan mock bisa diaplikasikan
_flask_app = None

def get_flask_app():
    global _flask_app
    if _flask_app is None:
        # Mock YOLO sebelum 'neopark_server' diimpor pertama kali
        # Ini penting jika YOLO diinisialisasi secara global di neopark_server.py
        mock_yolo_instance = MagicMock()
        # Anda bisa mengatur perilaku mock_yolo_instance di sini jika perlu
        # Misalnya: mock_yolo_instance.return_value = [MagicMock(boxes=...)]

        # Patch class YOLO di modul ultralytics yang mungkin diimpor oleh neopark_server
        # Atau patch langsung di 'neopark_server.YOLO' jika itu cara pemanggilannya
        with patch('ultralytics.YOLO', return_value=mock_yolo_instance) as mock_yolo_class_ultralytics, \
             patch('neopark_server.YOLO', return_value=mock_yolo_instance, create=True) as mock_yolo_class_server:
            # 'create=True' untuk neopark_server.YOLO jika YOLO diimpor dan diassign di sana
            
            # Simpan mock class untuk digunakan di tes jika perlu
            get_flask_app.mock_yolo_class_ultralytics = mock_yolo_class_ultralytics
            get_flask_app.mock_yolo_class_server = mock_yolo_class_server
            get_flask_app.mock_yolo_instance = mock_yolo_instance

            from neopark_server import app as actual_flask_app
            _flask_app = actual_flask_app
    return _flask_app


@pytest.fixture(scope='session') # Scope session agar mock YOLO konsisten
def app_instance():
    """Fixture untuk instance aplikasi Flask dengan YOLO di-mock."""
    flask_app = get_flask_app()
    flask_app.config.update({
        "TESTING": True,
        # Nonaktifkan logging Prometheus saat testing agar tidak pollute output / error jika endpoint tidak siap
        "PROMETHEUS_ENABLE_MIDDLEWARE": False,
    })
    yield flask_app

@pytest.fixture
def client(app_instance):
    """Fixture untuk Flask test client."""
    with app_instance.test_client() as client:
        yield client

@pytest.fixture
def runner(app_instance):
    """Fixture untuk Flask CLI runner."""
    return app_instance.test_cli_runner()

@pytest.fixture(autouse=True) # autouse=True agar fixture ini otomatis dipakai di setiap tes
def mock_prom_metrics_on_app(app_instance, monkeypatch):
    """
    Monkeypatch Prometheus metrics Gauges and Histograms
    untuk menghindari state antar tes dan error jika metrik sudah terdaftar.
    Ini juga mencegah panggilan .observe() atau .set() sungguhan ke Prometheus client.
    """
    # Mock Gauge
    mock_gauge_instance = MagicMock()
    monkeypatch.setattr('neopark_server.Gauge', MagicMock(return_value=mock_gauge_instance))
    
    # Mock Histogram
    mock_histogram_instance = MagicMock()
    # mock_histogram_instance.labels.return_value.observe = MagicMock() # Jika perlu lebih spesifik
    monkeypatch.setattr('neopark_server.Histogram', MagicMock(return_value=mock_histogram_instance))

    # Mock Counter
    mock_counter_instance = MagicMock()
    monkeypatch.setattr('neopark_server.Counter', MagicMock(return_value=mock_counter_instance))
    
    # Simpan mock untuk bisa diakses di tes jika perlu
    app_instance.prom_gauge_mock = mock_gauge_instance
    app_instance.prom_histogram_mock = mock_histogram_instance
    app_instance.prom_counter_mock = mock_counter_instance


@pytest.fixture
def clean_areas_data(app_instance):
    """Fixture untuk membersihkan/mereset areas_data sebelum setiap tes."""
    # Impor areas_data dari modul aplikasi yang sudah di-patch
    from neopark_server import areas_data
    
    original_areas_data = {
        area: data.copy() for area, data in areas_data.items()
    }
    
    # Reset ke state awal yang diketahui
    for area_id in areas_data:
        areas_data[area_id]['latest_detection'] = {}
        areas_data[area_id]['latest_frame'] = None
        areas_data[area_id]['processed_frame'] = None
        areas_data[area_id]['last_frame_time'] = None
        areas_data[area_id]['connection_status'] = False
        # Jangan reset 'frame_lock' karena itu objek threading.Lock

    yield areas_data # Berikan areas_data yang bersih ke tes

    # Kembalikan ke state original setelah tes (opsional, tapi baik untuk isolasi)
    # Namun, karena ini global, lebih aman meresetnya di awal setiap tes.
    # Jika ada tes yang berjalan paralel, ini bisa jadi masalah.
    # Untuk sekarang, kita reset di awal.