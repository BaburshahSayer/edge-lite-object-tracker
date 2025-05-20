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
#output_folder = "output_frames"
#os.makedirs(output_folder, exist_ok=True)

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
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    detector = ObjectDetector(model_path="/Users/babursayer/Desktop/0101/edge-lite-object-tracker/models/yolov5s.onnx")
    tracker = Sort(max_age=10, min_hits=3)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_filename = "output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_filename, fourcc, 30.0, (frame_width, frame_height))

    print("🎥 Running detection and tracking. Press 'q' to quit.")
    frame_count = 0
    frame_skip = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        frame_count += 1
        frame_input = cv2.resize(frame, (640, 640))

        if frame_count % frame_skip == 0:
            start_time = time.time()
            detections = detector.detect(frame_input)

            # Filter only human (person)
            person_detections = [d for d in detections if d["label"] == "person"]

            track_detections = [{"bbox": det["bbox"], "label": det["label"]} for det in person_detections]
            tracked_objects_raw = tracker.update(track_detections)
            tracked_objects = match_detections_with_tracks(tracked_objects_raw, person_detections)

            # Track lifecycle info
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

            # Draw and show frame
            frame_output = draw_detections(frame_input.copy(), tracked_objects)
            out.write(frame_output)

            fps = 1.0 / (time.time() - start_time) if start_time > 0 else 0
            cv2.putText(frame_output, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 👥 Person count
            person_count = len([obj for obj in tracked_objects if obj["label"] == "person"])
            print(f"👤 Detected persons: {person_count}")

            # 📺 Advertisement display
            if person_count == 1:
                ad_image = cv2.imread("single_ad.jpg")
                if ad_image is not None:
                    ad_image = cv2.resize(ad_image, (640, 480))
                    cv2.imshow("Advertisement", ad_image)
                else:
                    print("⚠️ Could not load single_ad.jpg")
            elif person_count > 1:
                ad_image = cv2.imread("group_ad.jpg")
                if ad_image is not None:
                    ad_image = cv2.resize(ad_image, (640, 480))
                    cv2.imshow("Advertisement", ad_image)
                else:
                    print("⚠️ Could not load group_ad.jpg")
            else:
                cv2.destroyWindow("Advertisement")
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
