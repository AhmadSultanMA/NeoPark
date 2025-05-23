from flask import Flask, request, jsonify, Response
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import threading
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model
model = YOLO("fine-best.pt")

# Data structure to store information for both areas
areas_data = {
    'A1': {
        'latest_detection': {},
        'latest_frame': None,
        'processed_frame': None,
        'last_frame_time': None,
        'connection_status': False,
        'frame_lock': threading.Lock()
    },
    'A2': {
        'latest_detection': {},
        'latest_frame': None,
        'processed_frame': None,
        'last_frame_time': None,
        'connection_status': False,
        'frame_lock': threading.Lock()
    }
}

# Initialize the Flask app
app = Flask(__name__)

def process_image_for_area(area_id, img_bytes):
    """Process image for specific area"""
    area_data = areas_data[area_id]
    
    try:
        logger.info(f"Processing image for Area {area_id}: {len(img_bytes)} bytes")
        
        # Update connection status
        area_data['connection_status'] = True
        area_data['last_frame_time'] = datetime.now()
        
        # Store the latest frame
        with area_data['frame_lock']:
            area_data['latest_frame'] = img_bytes
        
        # Convert to PIL Image for processing
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Run YOLOv8 inference
        results = model(img)

        car_detections = []

        # Create a copy for drawing bounding boxes
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        # Iterate over results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name == 'car':
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        
                        # Draw bounding box
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                        
                        # Draw area label and detection info
                        label = f"Area {area_id} - Car: {conf:.2f}"
                        try:
                            font = ImageFont.truetype("arial.ttf", 56)
                            draw.text((x1, y1-60), f"Area {area_id}", fill="blue", font=font)
                            draw.text((x1, y1-20), f"Car: {conf:.2f}", fill="red", font=font)
                        except:
                            draw.text((x1, y1-60), f"Area {area_id}", fill="blue")
                            draw.text((x1, y1-20), f"Car: {conf:.2f}", fill="red")
                        
                        car_detections.append({
                            "class": class_name,
                            "confidence": conf,
                            "bounding_box": [x1, y1, x2, y2],
                            "area": area_id
                        })

        # Convert processed image back to bytes
        img_byte_arr = io.BytesIO()
        img_with_boxes.save(img_byte_arr, format='JPEG', quality=85)
        
        with area_data['frame_lock']:
            area_data['processed_frame'] = img_byte_arr.getvalue()

        area_data['latest_detection'] = {"detections": car_detections}
        
        logger.info(f"Area {area_id}: Found {len(car_detections)} cars")
        return {"status": "Image processed", "detections": car_detections, "area": area_id}
        
    except Exception as e:
        logger.error(f"Error processing image for Area {area_id}: {str(e)}")
        raise e

# Routes for Area A1
@app.route('/a1/upload', methods=['POST'])
def upload_image_a1():
    if not request.data:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        result = process_image_for_area('A1', request.data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/a1/get_detections', methods=['GET'])
def get_detections_a1():
    return get_detections_for_area('A1')

@app.route('/a1/status', methods=['GET'])
def get_status_a1():
    return get_status_for_area('A1')

@app.route('/a1/video_feed')
def video_feed_a1():
    return Response(generate_frames_for_area('A1'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/a1/raw_feed')
def raw_feed_a1():
    return Response(generate_raw_frames_for_area('A1'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Routes for Area A2
@app.route('/a2/upload', methods=['POST'])
def upload_image_a2():
    if not request.data:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        result = process_image_for_area('A2', request.data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/a2/get_detections', methods=['GET'])
def get_detections_a2():
    return get_detections_for_area('A2')

@app.route('/a2/status', methods=['GET'])
def get_status_a2():
    return get_status_for_area('A2')

@app.route('/a2/video_feed')
def video_feed_a2():
    return Response(generate_frames_for_area('A2'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/a2/raw_feed')
def raw_feed_a2():
    return Response(generate_raw_frames_for_area('A2'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Combined routes
@app.route('/combined/get_detections', methods=['GET'])
def get_combined_detections():
    """Get detections from both areas"""
    a1_data = get_area_detection_data('A1')
    a2_data = get_area_detection_data('A2')
    
    total_cars = a1_data['car_count'] + a2_data['car_count']
    
    return jsonify({
        "total_cars": total_cars,
        "area_a1": {
            "car_count": a1_data['car_count'],
            "detections": a1_data['detections'],
            "connection_status": a1_data['connection_status']
        },
        "area_a2": {
            "car_count": a2_data['car_count'],
            "detections": a2_data['detections'],
            "connection_status": a2_data['connection_status']
        },
        "confidence_threshold": 0.8
    })

@app.route('/combined/status', methods=['GET'])
def get_combined_status():
    """Get status from both areas"""
    return jsonify({
        "area_a1": get_status_data('A1'),
        "area_a2": get_status_data('A2')
    })

# Helper functions
def get_detections_for_area(area_id):
    """Get detections for specific area"""
    area_data = areas_data[area_id]
    
    # Check if we have recent data (within last 10 seconds)
    if area_data['last_frame_time']:
        time_diff = (datetime.now() - area_data['last_frame_time']).total_seconds()
        if time_diff > 10:
            area_data['connection_status'] = False
    
    if not area_data['latest_detection']:
        return jsonify({
            "status": "No detections yet",
            "object_counts": {"car": 0},
            "connection_status": area_data['connection_status'],
            "area": area_id
        })
    
    # Filter cars with confidence > 0.8
    high_confidence_cars = [
        d for d in area_data['latest_detection']["detections"] 
        if d["class"] == "car" and d["confidence"] > 0.8
    ]
    
    car_count = len(high_confidence_cars)
    
    return jsonify({
        "object_counts": {"car": car_count},
        "high_confidence_detections": high_confidence_cars,
        "total_detections": len(area_data['latest_detection']["detections"]),
        "confidence_threshold": 0.8,
        "connection_status": area_data['connection_status'],
        "last_update": area_data['last_frame_time'].isoformat() if area_data['last_frame_time'] else None,
        "area": area_id
    })

def get_status_for_area(area_id):
    """Get status for specific area"""
    area_data = areas_data[area_id]
    
    if area_data['last_frame_time']:
        time_diff = (datetime.now() - area_data['last_frame_time']).total_seconds()
        if time_diff > 10:
            area_data['connection_status'] = False
    
    return jsonify({
        "connection_status": area_data['connection_status'],
        "last_frame_time": area_data['last_frame_time'].isoformat() if area_data['last_frame_time'] else None,
        "has_frame": area_data['latest_frame'] is not None,
        "area": area_id
    })

def get_area_detection_data(area_id):
    """Get detection data for area (helper for combined route)"""
    area_data = areas_data[area_id]
    
    if area_data['last_frame_time']:
        time_diff = (datetime.now() - area_data['last_frame_time']).total_seconds()
        if time_diff > 10:
            area_data['connection_status'] = False
    
    if not area_data['latest_detection']:
        return {
            "car_count": 0,
            "detections": [],
            "connection_status": area_data['connection_status']
        }
    
    high_confidence_cars = [
        d for d in area_data['latest_detection']["detections"] 
        if d["class"] == "car" and d["confidence"] > 0.8
    ]
    
    return {
        "car_count": len(high_confidence_cars),
        "detections": high_confidence_cars,
        "connection_status": area_data['connection_status']
    }

def get_status_data(area_id):
    """Get status data for area (helper for combined route)"""
    area_data = areas_data[area_id]
    
    if area_data['last_frame_time']:
        time_diff = (datetime.now() - area_data['last_frame_time']).total_seconds()
        if time_diff > 10:
            area_data['connection_status'] = False
    
    return {
        "connection_status": area_data['connection_status'],
        "last_frame_time": area_data['last_frame_time'].isoformat() if area_data['last_frame_time'] else None,
        "has_frame": area_data['latest_frame'] is not None
    }

def generate_frames_for_area(area_id):
    """Generate video frames for specific area"""
    area_data = areas_data[area_id]
    
    while True:
        with area_data['frame_lock']:
            if area_data['processed_frame'] is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + area_data['processed_frame'] + b'\r\n')
            else:
                placeholder = create_placeholder_image(area_id)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        time.sleep(0.06)

def generate_raw_frames_for_area(area_id):
    """Generate raw video frames for specific area"""
    area_data = areas_data[area_id]
    
    while True:
        with area_data['frame_lock']:
            if area_data['latest_frame'] is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + area_data['latest_frame'] + b'\r\n')
            else:
                placeholder = create_placeholder_image(area_id)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        time.sleep(0.1)

def create_placeholder_image(area_id):
    """Create a placeholder image when camera is not connected"""
    img = Image.new('RGB', (640, 480), color='gray')
    draw = ImageDraw.Draw(img)
    
    text = f"Area {area_id} - Camera Disconnected"
    try:
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((640 - text_width) // 2, (480 - text_height) // 2)
        draw.text(position, text, fill="white", font=font)
    except:
        draw.text((200, 240), text, fill="white")
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

# CORS support for cross-origin requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    logger.info("Starting Combined Car Detection Server...")
    logger.info("Area A1 endpoints: /a1/upload, /a1/get_detections, /a1/video_feed, /a1/raw_feed")
    logger.info("Area A2 endpoints: /a2/upload, /a2/get_detections, /a2/video_feed, /a2/raw_feed")
    logger.info("Combined endpoints: /combined/get_detections, /combined/status")
    logger.info("Make sure your ESP32CAM devices are configured to send images to the correct endpoints")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)