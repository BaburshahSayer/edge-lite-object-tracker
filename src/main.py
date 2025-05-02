import cv2
from detector import ObjectDetector
from tracker import Sort
import time
from utils import match_detections_with_tracks


# Global flag to detect close button click
close_button_clicked = False

# Draw detection boxes and add track ID labels
def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det.get('label', 'object')} ID: {det.get('track_id', '?')}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return frame

# Draw a close button on the frame
def draw_close_button(frame):
    button_top_left = (10, 10)
    button_bottom_right = (100, 50)
    cv2.rectangle(frame, button_top_left, button_bottom_right, (0, 0, 255), -1)  # Red background
    cv2.putText(frame, "Close", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame, button_top_left, button_bottom_right

# Mouse callback function to detect button click
def mouse_callback(event, x, y, flags, param):
    global close_button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        button_top_left, button_bottom_right = param
        if (button_top_left[0] <= x <= button_bottom_right[0]) and \
           (button_top_left[1] <= y <= button_bottom_right[1]):
            print("Close button clicked!")
            close_button_clicked = True

def main():
    global close_button_clicked
    cap = cv2.VideoCapture(0)
    detector = ObjectDetector(model_path="/Users/babursayer/Desktop/0101/edge-lite-object-tracker/models/yolov5s.onnx") 
    tracker = Sort()

    print("Running detection and tracking. Press 'q' or click 'Close' to quit.")

    frame_skip = 3  # this is the frame skip while being track at the same time. it makes it bit faster while detection.
    frame_count = 0

    window_name = "EdgeLite Detection and Tracking"
    cv2.namedWindow(window_name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or close_button_clicked:
            break

        frame_count += 1
        frame_resized = cv2.resize(frame, (640, 480))

        # close button at the screen to make it easy to close it.
        frame_resized, btn_topleft, btn_bottomright = draw_close_button(frame_resized)
        cv2.setMouseCallback(window_name, mouse_callback, (btn_topleft, btn_bottomright))

        if frame_count % frame_skip != 0:
            cv2.imshow(window_name, frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Run object detection
        start_time = time.time()
        detections = detector.detect(frame_resized)
        detection_time = time.time() - start_time

        # Prepare detections for tracking
        track_detections = [{"bbox": det["bbox"], "label": det["label"]} for det in detections]
        tracked_objects_raw = tracker.update(track_detections)

        # Add label back to tracked objects by matching bbox
        tracked_objects = match_detections_with_tracks(tracked_objects_raw, detections)


        # Draw tracking results
        frame_resized = draw_detections(frame_resized, tracked_objects)

        # Show FPS
        fps = 1.0 / detection_time if detection_time > 0 else 0
        cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow(window_name, frame_resized)

        # Exit on key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
