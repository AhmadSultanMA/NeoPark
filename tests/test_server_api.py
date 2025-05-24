# tests/test_server_api.py
import json
from unittest.mock import patch, MagicMock, ANY
import io
import pytest # Untuk parametrize
from datetime import datetime

# Menggunakan fixture clean_areas_data secara otomatis
pytestmark = pytest.mark.usefixtures("clean_areas_data")

# --- Tes untuk /aX/status dan /combined/status ---
@pytest.mark.parametrize("area_id_param", ["A1", "A2"])
def test_get_status_area(client, area_id_param, clean_areas_data): # clean_areas_data untuk isolasi
    from neopark_server import areas_data # Akses areas_data yang bersih
    areas_data[area_id_param].update({
        'last_frame_time': datetime.now(),
        'latest_frame': b'frame',
        'connection_status': True
    })
    response = client.get(f'/{area_id_param.lower()}/status')
    assert response.status_code == 200
    data = response.json
    assert data['area'] == area_id_param
    assert data['connection_status'] is True
    assert data['has_frame'] is True

def test_get_combined_status_api(client, clean_areas_data):
    from neopark_server import areas_data
    areas_data["A1"]['connection_status'] = True
    areas_data["A2"]['connection_status'] = False
    areas_data["A1"]['latest_frame'] = b'frame'

    response = client.get('/combined/status')
    assert response.status_code == 200
    data = response.json
    assert data['area_a1']['connection_status'] is True
    assert data['area_a1']['has_frame'] is True
    assert data['area_a2']['connection_status'] is False
    assert data['area_a2']['has_frame'] is False


# --- Tes untuk /aX/upload ---
@pytest.mark.parametrize("area_id_param", ["A1", "A2"])
@patch('neopark_server.process_image_for_area') # Mock fungsi inti yang berat
def test_upload_image_area_success(mock_process_image, client, area_id_param, clean_areas_data):
    expected_response_from_mock = {
        "status": f"Image processed for {area_id_param} (mocked)",
        "detections": [{"class": "car", "confidence": 0.92, "area": area_id_param}],
        "area": area_id_param
    }
    mock_process_image.return_value = expected_response_from_mock
    
    dummy_image_bytes = b'TestDummyImageData'
    response = client.post(f'/{area_id_param.lower()}/upload', data=dummy_image_bytes)
    
    assert response.status_code == 200
    assert response.json == expected_response_from_mock
    mock_process_image.assert_called_once_with(area_id_param, dummy_image_bytes)


# --- Tes untuk /aX/get_detections dan /combined/get_detections ---
@pytest.mark.parametrize("area_id_param", ["A1", "A2"])
def test_get_detections_area_api(client, area_id_param, clean_areas_data):
    from neopark_server import areas_data
    # Setup data deteksi untuk area ini
    areas_data[area_id_param].update({
        'last_frame_time': datetime.now(),
        'connection_status': True,
        'latest_detection': {
            "detections": [
                {"class": "car", "confidence": 0.95, "area": area_id_param, "bounding_box": [1,2,3,4]},
                {"class": "car", "confidence": 0.70, "area": area_id_param, "bounding_box": [5,6,7,8]}
            ]
        }
    })
    response = client.get(f'/{area_id_param.lower()}/get_detections')
    assert response.status_code == 200
    data = response.json
    assert data['area'] == area_id_param
    assert data['object_counts']['car'] == 1 # Hanya yang > 0.8
    assert len(data['high_confidence_detections']) == 1
    assert data['high_confidence_detections'][0]['confidence'] == 0.95
    assert data['total_detections_in_frame'] == 2 # Semua deteksi di frame itu

def test_get_combined_detections_api(client, clean_areas_data):
    from neopark_server import areas_data
    # Setup A1
    areas_data["A1"].update({
        'last_frame_time': datetime.now(), 'connection_status': True,
        'latest_detection': {"detections": [{"class": "car", "confidence": 0.95, "area": "A1", "bounding_box": [1,2,3,4]}]}
    })
    # Setup A2
    areas_data["A2"].update({
        'last_frame_time': datetime.now(), 'connection_status': True,
        'latest_detection': {
            "detections": [
                {"class": "car", "confidence": 0.85, "area": "A2", "bounding_box": [5,6,7,8]},
                {"class": "car", "confidence": 0.90, "area": "A2", "bounding_box": [9,10,11,12]}
            ]
        }
    })
    response = client.get('/combined/get_detections')
    assert response.status_code == 200
    data = response.json
    assert data['total_cars'] == 3 # 1 dari A1 + 2 dari A2 (semua > 0.8)
    assert data['area_a1']['car_count'] == 1
    assert data['area_a2']['car_count'] == 2


# --- Tes untuk Video Feed Endpoints (Contoh Dasar) ---
@pytest.mark.parametrize("area_id_param", ["A1", "A2"])
@patch('neopark_server.generate_frames_for_area') # Mock generatornya
def test_video_feed_area(mock_generate_frames, client, area_id_param, clean_areas_data):
    # Buat mock generator mengembalikan frame sederhana
    def dummy_frame_generator(area_id_gen):
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'dummyjpegdata' + b'\r\n')
        # yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + create_placeholder_image(area_id_gen) + b'\r\n')


    mock_generate_frames.side_effect = dummy_frame_generator # Gunakan side_effect untuk generator
    
    response = client.get(f'/{area_id_param.lower()}/video_feed')
    assert response.status_code == 200
    assert response.mimetype == 'multipart/x-mixed-replace'
    
    mock_generate_frames.assert_called_once_with(area_id_param)

# --- Tes untuk /metrics endpoint ---
def test_metrics_endpoint_more_details(client, clean_areas_data):
    response = client.get('/metrics')
    assert response.status_code == 200
    assert 'text/plain' in response.content_type
    metrics_text = response.data.decode('utf-8')
    
    # Cek beberapa metrik custom dan standar
    assert 'neopark_occupied_slots_area_a1' in metrics_text
    assert 'neopark_occupied_slots_area_a2' in metrics_text
    assert 'neopark_yolo_detection_confidence_score_histogram_bucket' in metrics_text
    assert 'neopark_yolo_car_detections_total' in metrics_text
    assert 'flask_http_requests_total' in metrics_text # Dari prometheus_flask_exporter
    assert 'flask_http_request_latency_seconds_bucket' in metrics_text # atau _duration_seconds