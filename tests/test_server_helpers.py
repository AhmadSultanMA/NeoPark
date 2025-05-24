# tests/test_server_helpers.py
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from PIL import Image
import io

# Impor dari server; pastikan sys.path di conftest.py sudah benar
from neopark_server import (
    create_placeholder_image,
    get_status_data,
    get_area_detection_data,
    get_detections_for_area, # Ini mengembalikan jsonify, jadi perlakuannya beda
    process_image_for_area, # Kita akan mock bagian YOLO-nya
    areas_data # Untuk setup data tes
)
import neopark_server # Untuk mengakses mock Prometheus dari app_instance

# Menggunakan fixture clean_areas_data secara otomatis untuk setiap tes di file ini
pytestmark = pytest.mark.usefixtures("clean_areas_data")


# --- Tes untuk create_placeholder_image ---
@pytest.mark.parametrize("area_id_param", ["A1", "X99"])
def test_create_placeholder_image_valid_jpeg(area_id_param):
    img_bytes = create_placeholder_image(area_id_param)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 0
    try:
        img = Image.open(io.BytesIO(img_bytes))
        assert img.format == "JPEG"
        assert img.size == (640, 480) # Asumsi ukuran placeholder tetap
    except Exception as e:
        pytest.fail(f"Placeholder image bukan JPEG valid atau ukuran salah: {e}")

# --- Tes untuk get_status_data ---
def test_get_status_data_default_disconnected(clean_areas_data): # clean_areas_data di sini untuk menegaskan penggunaan
    status = get_status_data("A1")
    assert status['connection_status'] is False
    assert status['has_frame'] is False
    assert status['last_frame_time'] is None

def test_get_status_data_connected(clean_areas_data):
    areas_data["A1"].update({
        'last_frame_time': datetime.now(),
        'latest_frame': b'some_frame_data',
        'connection_status': True # Biasanya di-set oleh process_image_for_area
    })
    status = get_status_data("A1")
    assert status['connection_status'] is True
    assert status['has_frame'] is True
    assert status['last_frame_time'] is not None

def test_get_status_data_timeout_updates_status(clean_areas_data):
    areas_data["A1"].update({
        'last_frame_time': datetime.now() - timedelta(seconds=20), # Timeout
        'latest_frame': b'some_frame_data',
        'connection_status': True # Awalnya True
    })
    # Panggil fungsi yang punya side-effect mengubah status jika timeout
    from neopark_server import get_status_for_area # Import fungsi endpoint
    _ = get_status_for_area("A1") # Panggil untuk memicu logika timeout internalnya

    status_after_call = get_status_data("A1") # Ambil status bersih setelahnya
    assert status_after_call['connection_status'] is False # Harusnya sudah False


# --- Tes untuk get_area_detection_data (helper untuk /combined/get_detections) ---
def test_get_area_detection_data_no_detections(clean_areas_data):
    result = get_area_detection_data("A1")
    assert result["car_count"] == 0
    assert result["detections"] == []
    assert result["connection_status"] is False

def test_get_area_detection_data_with_mixed_confidence(clean_areas_data):
    areas_data["A1"].update({
        'last_frame_time': datetime.now(),
        'connection_status': True,
        'latest_detection': {
            "detections": [
                {"class": "car", "confidence": 0.95, "area": "A1", "bounding_box": [1,2,3,4]},
                {"class": "car", "confidence": 0.75, "area": "A1", "bounding_box": [5,6,7,8]}, # Di bawah threshold 0.8
                {"class": "person", "confidence": 0.90, "area": "A1", "bounding_box": [9,10,11,12]},
                {"class": "car", "confidence": 0.85, "area": "A1", "bounding_box": [13,14,15,16]}
            ]
        }
    })
    result = get_area_detection_data("A1")
    assert result["car_count"] == 2 # Hanya yang confidence > 0.8
    assert len(result["detections"]) == 2
    assert result["detections"][0]["confidence"] == 0.95
    assert result["detections"][1]["confidence"] == 0.85
    assert result["connection_status"] is True


def test_process_image_for_area_basic_flow(app_instance, clean_areas_data):
    mock_model = get_flask_app.mock_yolo_instance # Ambil instance mock model dari conftest

    mock_result_box = MagicMock()
    mock_result_box.xyxy = [[10, 20, 30, 40]] # Koordinat bounding box
    mock_result_box.conf = [0.95] # Confidence score
    mock_result_box.cls = [0] # Class ID (misalnya 0 untuk 'car')

    mock_yolo_result = MagicMock()
    mock_yolo_result.boxes = [mock_result_box] # Satu deteksi
    mock_model.return_value = [mock_yolo_result] # model(img) akan mengembalikan list ini

    mock_model.names = {0: 'car'}

    # Buat gambar dummy JPEG
    dummy_jpeg_bytes = create_placeholder_image("dummy_for_processing")

    result_dict = process_image_for_area("A1", dummy_jpeg_bytes)

    assert result_dict["status"] == "Image processed"
    assert result_dict["area"] == "A1"
    assert len(result_dict["detections"]) == 1
    assert result_dict["detections"][0]["class"] == "car"
    assert result_dict["detections"][0]["confidence"] == 0.95

    # Verifikasi metrik Prometheus di-update (via mock dari conftest)
    app_instance.prom_gauge_mock.set.assert_any_call(1) # occupied_slots_a1.set(1)
    app_instance.prom_histogram_mock.labels.assert_called_with(area="A1")
    app_instance.prom_histogram_mock.labels.return_value.observe.assert_called_with(0.95)
    app_instance.prom_counter_mock.labels.assert_called_with(area="A1")
    app_instance.prom_counter_mock.labels.return_value.inc.assert_called_once()

    assert areas_data["A1"]["connection_status"] is True
    assert areas_data["A1"]["processed_frame"] is not None