# tests/test_server_helpers.py
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from PIL import Image
import io
import os # Untuk path operations

# Impor dari server
from neopark_server import (
    create_placeholder_image,
    get_status_data,
    get_area_detection_data,
    process_image_for_area,
    areas_data
)
# Kita tidak lagi butuh 'import neopark_server' hanya untuk app_instance.prom_...
# karena kita akan akses mock dari app_instance langsung jika dilampirkan di conftest.

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
        assert img.size == (640, 480)
    except Exception as e:
        pytest.fail(f"Placeholder image bukan JPEG valid atau ukuran salah: {e}")

# --- Tes untuk get_status_data ---
def test_get_status_data_default_disconnected(clean_areas_data):
    status = get_status_data("A1")
    assert status['connection_status'] is False
    assert status['has_frame'] is False
    assert status['last_frame_time'] is None

def test_get_status_data_connected(clean_areas_data):
    areas_data["A1"].update({
        'last_frame_time': datetime.now(),
        'latest_frame': b'some_frame_data',
        'connection_status': True
    })
    status = get_status_data("A1")
    assert status['connection_status'] is True
    assert status['has_frame'] is True
    assert status['last_frame_time'] is not None

def test_get_status_data_timeout_updates_status(clean_areas_data):
    areas_data["A1"].update({
        'last_frame_time': datetime.now() - timedelta(seconds=20),
        'latest_frame': b'some_frame_data',
        'connection_status': True
    })
    from neopark_server import get_status_for_area # Fungsi endpoint yang memicu logika
    _ = get_status_for_area("A1")
    status_after_call = get_status_data("A1")
    assert status_after_call['connection_status'] is False


# --- Tes untuk get_area_detection_data ---
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
                {"class": "car", "confidence": 0.75, "area": "A1", "bounding_box": [5,6,7,8]},
                {"class": "person", "confidence": 0.90, "area": "A1", "bounding_box": [9,10,11,12]},
                {"class": "car", "confidence": 0.85, "area": "A1", "bounding_box": [13,14,15,16]}
            ]
        }
    })
    result = get_area_detection_data("A1")
    assert result["car_count"] == 2
    assert len(result["detections"]) == 2
    assert result["detections"][0]["confidence"] == 0.95
    assert result["detections"][1]["confidence"] == 0.85
    assert result["connection_status"] is True


# --- Tes untuk process_image_for_area ---
def test_process_image_for_area_with_sample_image(app_instance, clean_areas_data):
    # Akses mock model yang sudah di-attach ke app_instance di conftest.py
    # Pastikan 'app_instance.mock_yolo_instance' sudah di-set di conftest.py
    assert hasattr(app_instance, 'mock_yolo_instance'), "Mock YOLO instance tidak ditemukan di app_instance. Periksa conftest.py"
    mock_model = app_instance.mock_yolo_instance

    mock_result_box = MagicMock()
    mock_result_box.xyxy = [[10, 20, 30, 40]]
    mock_result_box.conf = [0.95]
    mock_result_box.cls = [0] # Misal 0 adalah 'car'

    mock_yolo_result = MagicMock()
    mock_yolo_result.boxes = [mock_result_box]
    mock_model.return_value = [mock_yolo_result] # model(img) akan mengembalikan ini
    mock_model.names = {0: 'car'} # Pastikan mapping class ID ke nama class ada

    test_dir = os.path.dirname(os.path.abspath(__file__))
    sample_image_path = os.path.join(test_dir, "sample_images", "one_car.jpg")

    # Pastikan file gambar sampel ada, jika tidak, skip tes ini atau beri pesan error
    if not os.path.exists(sample_image_path):
        pytest.skip(f"File gambar sampel tidak ditemukan: {sample_image_path}")

    with open(sample_image_path, "rb") as f:
        image_bytes_from_file = f.read()
    # --- Selesai memuat gambar ---

    # Jalankan fungsi yang diuji
    result_dict = process_image_for_area("A1", image_bytes_from_file)

    # Asersi tetap sama, karena hasil deteksi dikontrol oleh mock_model
    assert result_dict["status"] == "Image processed"
    assert result_dict["area"] == "A1"
    assert len(result_dict["detections"]) == 1
    detections = result_dict["detections"]
    assert detections[0]["class"] == "car"
    assert detections[0]["confidence"] == 0.95

    # Verifikasi metrik Prometheus di-update (via mock dari conftest)
    # Akses mock yang sudah di-attach ke app_instance di conftest.py
    app_instance.prom_gauge_mock.set.assert_any_call(1)
    app_instance.prom_histogram_mock.labels.assert_called_with(area="A1")
    app_instance.prom_histogram_mock.labels.return_value.observe.assert_called_with(0.95)
    app_instance.prom_counter_mock.labels.assert_called_with(area="A1")
    app_instance.prom_counter_mock.labels.return_value.inc.assert_called_once()

    assert areas_data["A1"]["connection_status"] is True
    assert areas_data["A1"]["processed_frame"] is not None
    # Anda juga bisa mencoba membuka areas_data["A1"]["processed_frame"] dengan Pillow
    # untuk memastikan itu adalah gambar JPEG yang valid, sebagai asersi tambahan.
    try:
        processed_img = Image.open(io.BytesIO(areas_data["A1"]["processed_frame"]))
        assert processed_img.format == "JPEG"
    except Exception as e:
        pytest.fail(f"Processed frame bukan JPEG valid: {e}")