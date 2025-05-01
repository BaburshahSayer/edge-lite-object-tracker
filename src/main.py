import cv2
from detector import ObjectDetector
from tracker import Sort
import time

# Draw detection boxes and add a close button
def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"ID: {det['track_id']}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return frame

# Function to draw and detect button
def draw_close_button(frame):
    # Define button dimensions
    button_top_left = (10, 10)
    button_bottom_right = (100, 50)
    
    # Draw a rectangle for the button
    cv2.rectangle(frame, button_top_left, button_bottom_right, (0, 0, 255), -1)  # Red color
    
    # Put text inside the button
    cv2.putText(frame, "Close", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, button_top_left, button_bottom_right


def main():
    cap = cv2.VideoCapture(0)  # Use webcam (change to file path for video)
    detector = ObjectDetector(model_path="models/yolov5n.onnx")
    tracker = Sort()

    print("Running detection and tracking. Press 'q' to quit.")
    
    frame_skip = 1  # Skip every nth frame to improve FPS
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Skip frames for optimization (adjust frame_skip value)
        if frame_count % frame_skip != 0:
            continue

        # Step 1: Resize frame for faster processing (optional)
        frame_resized = cv2.resize(frame, (640, 480))

        # Step 2: Run Detection
        start_time = time.time()
        detections = detector.detect(frame_resized)
        print("Detections:", detections)  # <-- Add this line
        detection_time = time.time() - start_time

        # Prepare detections for tracking
        track_detections = [{
            "bbox": det["bbox"]
        } for det in detections]

        # Update tracker
        tracked_objects = tracker.update(track_detections)


        # Step 3: Update Tracker
        [{"bbox": [...], "track_id": 1}, ...]

        # Step 4: Draw detections and track IDs
        frame_resized = draw_detections(frame_resized, tracked_objects)

        # Step 5: Display result with FPS
        fps = 1.0 / detection_time
        cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("EdgeLite Detection and Tracking", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
