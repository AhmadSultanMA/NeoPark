from flask import Flask, request, jsonify, Response
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import threading
import time
import logging
from datetime import datetime

from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Gauge, Histogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE_PATH = "fine-best.pt"
_model_instance = None


def get_yolo_model():
    global _model_instance
    if _model_instance is None:
        try:
            logger.info(f"Attempting to load YOLO model from: {MODEL_FILE_PATH}")
            _model_instance = YOLO(MODEL_FILE_PATH)
            logger.info("YOLO model loaded successfully for application runtime.")
        except Exception as e:
            logger.error(
                f"FATAL: Failed to load YOLO model from {MODEL_FILE_PATH}: {e}"
            )
            raise RuntimeError(f"Could not load YOLO model: {e}")
    return _model_instance


areas_data = {
    "A1": {
        "latest_detection": {},
        "latest_frame": None,
        "processed_frame": None,
        "last_frame_time": None,
        "connection_status": False,
        "frame_lock": threading.Lock(),
    },
    "A2": {
        "latest_detection": {},
        "latest_frame": None,
        "processed_frame": None,
        "last_frame_time": None,
        "connection_status": False,
        "frame_lock": threading.Lock(),
    },
}

app = Flask(__name__)

metrics = PrometheusMetrics(app, group_by="endpoint")

occupied_slots_a1 = Gauge(
    "neopark_occupied_slots_area_a1", "Number of occupied parking slots in Area A1"
)
occupied_slots_a2 = Gauge(
    "neopark_occupied_slots_area_a2", "Number of occupied parking slots in Area A2"
)

yolo_confidence_scores = Histogram(
    "neopark_yolo_detection_confidence_score_histogram",
    "Histogram of YOLO detection confidence scores for detected cars",
    ["area"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

yolo_car_detections_total = Counter(
    "neopark_yolo_car_detections_total",
    "Total number of car detections by YOLO model",
    ["area"],
)


def process_image_for_area(area_id, img_bytes):
    area_data = areas_data[area_id]
    model_to_use = get_yolo_model()

    try:
        logger.info(f"Processing image for Area {area_id}: {len(img_bytes)} bytes")

        area_data["connection_status"] = True
        area_data["last_frame_time"] = datetime.now()

        with area_data["frame_lock"]:
            area_data["latest_frame"] = img_bytes

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        results = model_to_use(img)
        car_detections_list = []
        num_cars_in_frame = 0

        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model_to_use.names[class_id]
                    if class_name == "car":
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])

                        if conf > 0.8:
                            num_cars_in_frame += 1
                            yolo_confidence_scores.labels(area=area_id).observe(conf)
                            yolo_car_detections_total.labels(area=area_id).inc()

                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            label_text_area = f"Area {area_id}"
                            label_text_car = f"Car: {conf:.2f}"
                            try:
                                font = ImageFont.truetype("arial.ttf", 36)
                                draw.text(
                                    (x1, y1 - 55),
                                    label_text_area,
                                    fill="blue",
                                    font=font,
                                )
                                draw.text(
                                    (x1, y1 - 35), label_text_car, fill="red", font=font
                                )
                            except IOError:
                                draw.text((x1, y1 - 30), label_text_area, fill="blue")
                                draw.text((x1, y1 - 10), label_text_car, fill="red")

                            car_detections_list.append(
                                {
                                    "class": class_name,
                                    "confidence": conf,
                                    "bounding_box": [x1, y1, x2, y2],
                                    "area": area_id,
                                }
                            )
        print(
            f"DEBUG: Final num_cars_in_frame for area {area_id} before set: {num_cars_in_frame}"
        )  # DEBUG
        if area_id == "A1":
            print(f"DEBUG: Calling occupied_slots_a1.set({num_cars_in_frame})")  # DEBUG
            occupied_slots_a1.set(num_cars_in_frame)
        elif area_id == "A2":
            print(f"DEBUG: Calling occupied_slots_a2.set({num_cars_in_frame})")  # DEBUG
            occupied_slots_a2.set(num_cars_in_frame)
        if area_id == "A1":
            occupied_slots_a1.set(num_cars_in_frame)
        elif area_id == "A2":
            occupied_slots_a2.set(num_cars_in_frame)

        img_byte_arr = io.BytesIO()
        img_with_boxes.save(img_byte_arr, format="JPEG", quality=85)

        with area_data["frame_lock"]:
            area_data["processed_frame"] = img_byte_arr.getvalue()

        area_data["latest_detection"] = {"detections": car_detections_list}

        logger.info(
            f"Area {area_id}: Found {num_cars_in_frame} cars for occupancy metric."
        )
        return {
            "status": "Image processed",
            "detections": car_detections_list,
            "area": area_id,
        }

    except Exception as e:
        logger.error(f"Error processing image for Area {area_id}: {str(e)}")
        raise e


@app.route("/a1/upload", methods=["POST"])
def upload_image_a1():
    if not request.data:
        return jsonify({"error": "No image data provided"}), 400
    try:
        result = process_image_for_area("A1", request.data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/a1/get_detections", methods=["GET"])
def get_detections_a1():
    return get_detections_for_area("A1")


@app.route("/a1/status", methods=["GET"])
def get_status_a1():
    return get_status_for_area("A1")


@app.route("/a1/video_feed")
@metrics.do_not_track()
def video_feed_a1():
    return Response(
        generate_frames_for_area("A1"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/a1/raw_feed")
@metrics.do_not_track()
def raw_feed_a1():
    return Response(
        generate_raw_frames_for_area("A1"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/a2/upload", methods=["POST"])
def upload_image_a2():
    if not request.data:
        return jsonify({"error": "No image data provided"}), 400
    try:
        result = process_image_for_area("A2", request.data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/a2/get_detections", methods=["GET"])
def get_detections_a2():
    return get_detections_for_area("A2")


@app.route("/a2/status", methods=["GET"])
def get_status_a2():
    return get_status_for_area("A2")


@app.route("/a2/video_feed")
@metrics.do_not_track()
def video_feed_a2():
    return Response(
        generate_frames_for_area("A2"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/a2/raw_feed")
@metrics.do_not_track()
def raw_feed_a2():
    return Response(
        generate_raw_frames_for_area("A2"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/combined/get_detections", methods=["GET"])
def get_combined_detections():
    a1_data = get_area_detection_data("A1")
    a2_data = get_area_detection_data("A2")
    total_cars = a1_data["car_count"] + a2_data["car_count"]
    return jsonify(
        {
            "total_cars": total_cars,
            "area_a1": {
                "car_count": a1_data["car_count"],
                "detections": a1_data["detections"],
                "connection_status": a1_data["connection_status"],
            },
            "area_a2": {
                "car_count": a2_data["car_count"],
                "detections": a2_data["detections"],
                "connection_status": a2_data["connection_status"],
            },
            "confidence_threshold": 0.8,
        }
    )


@app.route("/combined/status", methods=["GET"])
def get_combined_status():
    return jsonify({"area_a1": get_status_data("A1"), "area_a2": get_status_data("A2")})


def get_detections_for_area(area_id):
    area_data = areas_data[area_id]
    if area_data["last_frame_time"]:
        time_diff = (datetime.now() - area_data["last_frame_time"]).total_seconds()
        if time_diff > 10:
            area_data["connection_status"] = False
    if not area_data["latest_detection"] or not area_data["latest_detection"].get(
        "detections"
    ):
        return jsonify(
            {
                "status": "No detections yet",
                "object_counts": {"car": 0},
                "connection_status": area_data["connection_status"],
                "area": area_id,
            }
        )
    high_confidence_cars = [
        d
        for d in area_data["latest_detection"]["detections"]
        if d["class"] == "car" and d["confidence"] > 0.8
    ]
    car_count_high_conf = len(high_confidence_cars)
    return jsonify(
        {
            "object_counts": {"car": car_count_high_conf},
            "high_confidence_detections": high_confidence_cars,
            "total_detections_in_frame": len(
                area_data["latest_detection"]["detections"]
            ),
            "confidence_threshold": 0.8,
            "connection_status": area_data["connection_status"],
            "last_update": area_data["last_frame_time"].isoformat()
            if area_data["last_frame_time"]
            else None,
            "area": area_id,
        }
    )


def get_status_for_area(area_id):
    area_data = areas_data[area_id]
    if area_data["last_frame_time"]:
        time_diff = (datetime.now() - area_data["last_frame_time"]).total_seconds()
        if time_diff > 10:
            area_data["connection_status"] = False
    return jsonify(
        {
            "connection_status": area_data["connection_status"],
            "last_frame_time": area_data["last_frame_time"].isoformat()
            if area_data["last_frame_time"]
            else None,
            "has_frame": area_data["latest_frame"] is not None,
            "area": area_id,
        }
    )


def get_area_detection_data(area_id):
    area_data = areas_data[area_id]
    if area_data["last_frame_time"]:
        time_diff = (datetime.now() - area_data["last_frame_time"]).total_seconds()
        if time_diff > 10:
            area_data["connection_status"] = False
    if not area_data["latest_detection"] or not area_data["latest_detection"].get(
        "detections"
    ):
        return {
            "car_count": 0,
            "detections": [],
            "connection_status": area_data["connection_status"],
        }
    high_confidence_cars = [
        d
        for d in area_data["latest_detection"]["detections"]
        if d["class"] == "car" and d["confidence"] > 0.8
    ]
    return {
        "car_count": len(high_confidence_cars),
        "detections": high_confidence_cars,
        "connection_status": area_data["connection_status"],
    }


def get_status_data(area_id):
    area_data = areas_data[area_id]
    if area_data["last_frame_time"]:
        time_diff = (datetime.now() - area_data["last_frame_time"]).total_seconds()
        if time_diff > 10:
            area_data["connection_status"] = False
    return {
        "connection_status": area_data["connection_status"],
        "last_frame_time": area_data["last_frame_time"].isoformat()
        if area_data["last_frame_time"]
        else None,
        "has_frame": area_data["latest_frame"] is not None,
    }


def generate_frames_for_area(area_id):
    area_data = areas_data[area_id]
    while True:
        with area_data["frame_lock"]:
            if area_data["processed_frame"] is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + area_data["processed_frame"]
                    + b"\r\n"
                )
            else:
                placeholder = create_placeholder_image(area_id)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
                )
        time.sleep(0.06)


def generate_raw_frames_for_area(area_id):
    area_data = areas_data[area_id]
    while True:
        with area_data["frame_lock"]:
            if area_data["latest_frame"] is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + area_data["latest_frame"]
                    + b"\r\n"
                )
            else:
                placeholder = create_placeholder_image(area_id)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
                )
        time.sleep(0.1)


def create_placeholder_image(area_id):
    img = Image.new("RGB", (640, 480), color="gray")
    draw = ImageDraw.Draw(img)
    text = f"Area {area_id} - Camera Disconnected"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((640 - text_width) // 2, (480 - text_height) // 2)
        draw.text(position, text, fill="white", font=font)
    except IOError:
        draw.text((180, 230), text, fill="white")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE")
    return response


@app.route("/health", methods=["GET"])
@metrics.do_not_track()
def health_check():
    return (
        jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "neopark-server",
                "areas": {
                    "A1": {
                        "connection_status": areas_data["A1"]["connection_status"],
                        "has_frame": areas_data["A1"]["latest_frame"] is not None,
                    },
                    "A2": {
                        "connection_status": areas_data["A2"]["connection_status"],
                        "has_frame": areas_data["A2"]["latest_frame"] is not None,
                    },
                },
            }
        ),
        200,
    )


@app.route("/custom_metrics", methods=["GET"])
@metrics.do_not_track()
def custom_metrics_info():
    return (
        jsonify(
            {
                "available_metrics": [
                    "neopark_occupied_slots_area_a1",
                    "neopark_occupied_slots_area_a2",
                    "neopark_yolo_detection_confidence_score_histogram",
                    "neopark_yolo_car_detections_total",
                ],
                "metrics_endpoint": "/metrics",
                "note": "Access /metrics endpoint for Prometheus scraping",
            }
        ),
        200,
    )


if __name__ == "__main__":
    logger.info(
        "Starting Combined Car Detection Server with Prometheus metrics enabled on /metrics"
    )
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
