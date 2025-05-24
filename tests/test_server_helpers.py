# tests/test_server_helpers.py
import pytest
from datetime import datetime, timedelta
from PIL import Image
import io

# Impor dari server
from neopark_server import (
    create_placeholder_image,
    get_status_data,
    get_area_detection_data,
    areas_data,
    get_status_for_area,  # Dipindahkan ke impor utama
)

pytestmark = pytest.mark.usefixtures("clean_areas_data_fixture")


# --- Tes untuk create_placeholder_image ---
@pytest.mark.parametrize("area_id_param", ["A1", "X99"])
def test_create_placeholder_image_valid_jpeg(area_id_param):
    img_bytes = create_placeholder_image(area_id_param)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 0
    try:
        img = Image.open(io.BytesIO(img_bytes))
        assert img.format == "JPEG"
        assert img.size == (640, 480)  # Asumsi ukuran placeholder tetap
    except Exception as e:
        pytest.fail(f"Placeholder image bukan JPEG valid atau ukuran salah: {e}")


# --- Tes untuk get_status_data ---
def test_get_status_data_default_disconnected(clean_areas_data_fixture):
    status = get_status_data("A1")
    assert status["connection_status"] is False
    assert status["has_frame"] is False
    assert status["last_frame_time"] is None


def test_get_status_data_connected(clean_areas_data_fixture):
    areas_data["A1"].update(
        {
            "last_frame_time": datetime.now(),
            "latest_frame": b"some_frame_data",
            "connection_status": True,  # Biasanya di-set oleh process_image_for_area
        }
    )
    status = get_status_data("A1")
    assert status["connection_status"] is True
    assert status["has_frame"] is True
    assert status["last_frame_time"] is not None


def test_get_status_data_timeout_updates_status(app_instance, clean_areas_data_fixture):
    areas_data["A1"].update(
        {
            "last_frame_time": datetime.now() - timedelta(seconds=20),  # Timeout
            "latest_frame": b"some_frame_data",
            "connection_status": True,  # Awalnya True
        }
    )

    with app_instance.app_context():
        _ = get_status_for_area("A1")
    status_after_call = get_status_data("A1")
    assert status_after_call["connection_status"] is False


def test_get_area_detection_data_no_detections(clean_areas_data_fixture):
    result = get_area_detection_data("A1")
    assert result["car_count"] == 0
    assert result["detections"] == []
    assert result["connection_status"] is False


def test_get_area_detection_data_with_mixed_confidence(clean_areas_data_fixture):
    areas_data["A1"].update(
        {
            "last_frame_time": datetime.now(),
            "connection_status": True,
            "latest_detection": {
                "detections": [
                    {
                        "class": "car",
                        "confidence": 0.95,
                        "area": "A1",
                        "bounding_box": [1, 2, 3, 4],
                    },
                    {
                        "class": "car",
                        "confidence": 0.75,  # Di bawah threshold 0.8
                        "area": "A1",
                        "bounding_box": [5, 6, 7, 8],
                    },
                    {
                        "class": "person",
                        "confidence": 0.90,
                        "area": "A1",
                        "bounding_box": [9, 10, 11, 12],
                    },
                    {
                        "class": "car",
                        "confidence": 0.85,
                        "area": "A1",
                        "bounding_box": [13, 14, 15, 16],
                    },
                ]
            },
        }
    )
    result = get_area_detection_data("A1")
    assert result["car_count"] == 2  # Hanya yang confidence > 0.8 dan class 'car'
    assert len(result["detections"]) == 2
    assert result["detections"][0]["confidence"] == 0.95
    assert result["detections"][1]["confidence"] == 0.85
    assert result["connection_status"] is True
