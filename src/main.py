import cv2
import os
import time
import csv
from datetime import datetime
from detector import ObjectDetector
from tracker import Sort
from utils import match_detections_with_tracks

# Draw bounding boxes and labels
def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det.get('label', 'object')} ID: {det.get('track_id', '?')}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Folder to save image frames
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

# Track entry and exit times
object_lifecycle = {}

# Save tracking data to CSV
def save_tracking_data_to_csv(output_file="tracking_data.csv"):
    fieldnames = ["track_id", "label", "x1", "y1", "x2", "y2", "start_time", "end_time"]

    with open(output_file, mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for track_id, data in object_lifecycle.items():
            writer.writerow({
                "track_id": track_id,
                "label": data.get("label", "N/A"),
                "x1": data.get("bbox", [0, 0, 0, 0])[0],
                "y1": data.get("bbox", [0, 0, 0, 0])[1],
                "x2": data.get("bbox", [0, 0, 0, 0])[2],
                "y2": data.get("bbox", [0, 0, 0, 0])[3],
                "start_time": data.get("start_time", ""),
                "end_time": data.get("end_time", "")
            })

# Main function
def main():
    cap = cv2.VideoCapture(0)
    detector = ObjectDetector(model_path="/Users/babursayer/Desktop/0101/edge-lite-object-tracker/models/yolov5s.onnx")
    tracker = Sort(max_age=10, min_hits=3)

    # Save video as MJPEG (Motion JPEG)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_filename = "output_video.avi"  # Using .avi extension for MJPEG format
    
    # Use MJPG codec (Motion JPEG)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_filename, fourcc, 30.0, (frame_width, frame_height))

    print("Running detection and tracking. Press 'q' to quit.")
    frame_skip = 1
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_input = cv2.resize(frame, (640, 640))

        if frame_count % frame_skip == 0:
            start_time = time.time()
            detections = detector.detect(frame_input)
            detection_time = time.time() - start_time

            track_detections = []
            for det in detections:
                track_detections.append({
                    "bbox": det["bbox"],
                    "label": det["label"],
                })

            tracked_objects_raw = tracker.update(track_detections)
            tracked_objects = match_detections_with_tracks(tracked_objects_raw, detections)

            # Update object lifecycle
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            active_ids = set()

            for obj in tracked_objects:
                track_id = obj.get("track_id")
                active_ids.add(track_id)
                if track_id not in object_lifecycle:
                    object_lifecycle[track_id] = {
                        "label": obj.get("label", "N/A"),
                        "bbox": obj["bbox"],
                        "start_time": timestamp,
                        "end_time": timestamp
                    }
                else:
                    object_lifecycle[track_id]["bbox"] = obj["bbox"]
                    object_lifecycle[track_id]["end_time"] = timestamp

            frame_output = draw_detections(frame_input.copy(), tracked_objects)
            out.write(frame_output)

            fps = 1.0 / detection_time if detection_time > 0 else 0
            cv2.putText(frame_output, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            frame_output = frame_input.copy()

        # Save frame image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame_output)

        cv2.imshow("EdgeLite", frame_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    save_tracking_data_to_csv()

if __name__ == "__main__":
    main()
