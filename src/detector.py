import onnxruntime as ort
import numpy as np
import cv2

class ObjectDetector:
    def __init__(self, model_path):
        # Load the model using onnxruntime
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # YOLO class names (example for COCO dataset)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
            'backpack', 'umbrella', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed', 'N/A', 'dining table', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        input_array = np.array(blob)
        outputs = self.session.run([self.output_name], {self.input_name: input_array})[0]

        detections = []
        img_h, img_w = frame.shape[:2]

        for det in outputs[0]:  # assuming shape is [1, N, 85]
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] * det[4]  # class_conf * obj_conf

            if confidence > 0.4:
                cx, cy, w, h = det[0:4]
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                label = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "label": label
                })

        return detections
