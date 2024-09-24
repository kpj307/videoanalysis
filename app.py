import cv2
import tempfile
import math
import streamlit as st
from ultralytics import YOLO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# Load YOLOv8 Model
def load_yolo_model(model_path="yolov8m.pt"):
    model = YOLO(model_path)
    return model

# Detect objects in a frame
def detect_objects(frame, model):
    results = model(frame)  # YOLOv8 inference
    return results

# Process YOLOv8 detections and return detected objects in DeepSORT format
def process_detections(results, width, height):
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            obj_class = result.names[class_id]
            # DeepSORT expects detection as (x1, y1, width, height, confidence, class_name)
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, obj_class))
    return detections


# Log detected objects and events
def log_event(events_log, tracked_objects, timestamp):
    for obj in tracked_objects:
        event = {
            "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "object": obj['label'],
            "confidence": obj['confidence'],
            "box": obj['bbox'],
            "id": obj['tracker_id']  # Tracker ID from DeepSORT
        }
        events_log.append(event)

# Count objects by class and ID to ensure they're only counted once
def count_objects(tracked_objects, object_counts, tracked_ids):
    for obj in tracked_objects:
        obj_class = obj["label"]
        obj_id = obj["tracker_id"]

        if obj_id not in tracked_ids[obj_class]:
            tracked_ids[obj_class].add(obj_id)
            object_counts[obj_class] += 1

# Generate PDF report
def generate_pdf_report(filename, events_log, object_counts):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "CCTV Footage Analysis Report")
    y = 720

    # Write object counts to the report
    c.drawString(100, y, "Object Counts:")
    y -= 20
    for obj_class, count in object_counts.items():
        c.drawString(100, y, f"{obj_class}: {count}")
        y -= 20

    # Write detailed log of events to the report
    c.drawString(100, y, "Detected Events:")
    y -= 20
    for event in events_log:
        if event['confidence'] != None:
            c.drawString(100, y, f"{event['object']} detected | Confidence: {event['confidence']} | ID: {event['id']}")
        y -= 20
        if y < 100:
            c.showPage()  # Start a new page if too long
            y = 750
    c.save()

# Main function to analyze the video
def analyze_video(video_path, model_path="yolov8m.pt"):
    model = load_yolo_model(model_path)

    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=70)

    # Object counts and tracker IDs for each object type
    object_counts = defaultdict(int)
    tracked_ids = defaultdict(set)
    events_log = []

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Detect objects in the current frame
        results = detect_objects(frame, model)
        detections = process_detections(results, width, height)

        # Update DeepSORT tracker with the detections
        tracked_objects = tracker.update_tracks(detections, frame=frame)

        # Convert tracked objects into a list for logging and counting
        tracked_objects_data = []
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            tracked_objects_data.append({
                'tracker_id': track.track_id,
                'bbox': track.to_tlbr(),
                'label': track.get_det_class(),
                'confidence': track.get_det_conf()
            })

        # Log events with the current timestamp
        timestamp = datetime.now()
        log_event(events_log, tracked_objects_data, timestamp)

        # Count unique objects by their class and ID
        count_objects(tracked_objects_data, object_counts, tracked_ids)

    cap.release()

    # Create a temporary file for the report
    report_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    generate_pdf_report(report_file.name, events_log, object_counts)

    return report_file.name

# Streamlit App
st.title("Video with YOLOv8 and DeepSORT")

# File uploader for video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Store the uploaded video file temporarily
    video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(video_path, 'wb') as f:
        f.write(video_file.read())

    st.video(video_file)

    # Analyze the video and generate the report when the user clicks the button
    if st.button("Analyze Video"):
        st.write("Analyzing the video... This may take a while.")
        report_path = analyze_video(video_path)

        st.success("Analysis complete!")

        # Provide a download link for the generated report
        with open(report_path, "rb") as report_file:
            btn = st.download_button(
                label="Download Report",
                data=report_file,
                file_name="cctv_report.pdf",
                mime="application/pdf"
            )
