import onnxruntime as ort
import cv2
import numpy as np
import os

class ObjectDetector:
    def __init__(self, model_path):
        # Convert relative path to absolute path
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.join(project_root, model_path)
        
        print("Resolved model path:", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        self.session = ort.InferenceSession(model_path)

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1]  # BGR to RGB
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))
        return img.astype(np.float32)

    def detect(self, frame, conf_threshold=0.5):
        img = self.preprocess(frame)
        inputs = {self.session.get_inputs()[0].name: img}

        outputs = self.session.run(None, inputs)[0]

        detections = []
        for det in outputs[0]:
            x, y, w, h = det[:4]
            obj_conf = det[4]
            class_scores = det[5:]

            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            conf = obj_conf * class_conf

            if conf > conf_threshold:
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": int(class_id),
                    "label": str(class_id)
                })

        return detections
