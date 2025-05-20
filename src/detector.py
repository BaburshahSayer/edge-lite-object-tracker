import cv2
import numpy as np
import onnxruntime as ort
import time
from datetime import datetime

# ======= Helper Functions =======
def non_max_suppression(detections, iou_threshold=0.45):
    if len(detections) == 0:
        return []
    boxes = np.array([det["bbox"] for det in detections])
    scores = np.array([det["score"] for det in detections])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]

# ======= Object Detector =======
class ObjectDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = [
            'person'
        ]

    def detect(self, frame):
        input_size = 640
        original_h, original_w = frame.shape[:2]

        img_resized = cv2.resize(frame, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        outputs = self.session.run([self.output_name], {self.input_name: img})[0]
        detections = []

        for det in outputs[0]:
            obj_conf = det[4]
            class_scores = det[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            confidence = obj_conf * class_conf

            if confidence > 0.4 and class_id == 0:  # only keep 'person'
                cx, cy, w, h = det[:4]
                scale_x = original_w / input_size
                scale_y = original_h / input_size
                x1 = int((cx - w / 2) * scale_x)
                y1 = int((cy - h / 2) * scale_y)
                x2 = int((cx + w / 2) * scale_x)
                y2 = int((cy + h / 2) * scale_y)

                label = 'person'
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "score": confidence
                })

        return non_max_suppression(detections)

# ======= SORT Tracker =======
class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_id = 0
        self.tracks = {}

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            matched = False
            for tid, trk in self.tracks.items():
                if self.iou(det["bbox"], trk["bbox"]) > self.iou_threshold:
                    trk["bbox"] = det["bbox"]
                    trk["label"] = det["label"]
                    trk["last_seen"] = time.time()
                    trk["frames"].append(det["bbox"])
                    det["track_id"] = tid
                    matched = True
                    break
            if not matched:
                self.track_id += 1
                det["track_id"] = self.track_id
                self.tracks[self.track_id] = {
                    "bbox": det["bbox"],
                    "label": det["label"],
                    "start_time": datetime.now(),
                    "last_seen": time.time(),
                    "frames": [det["bbox"]]
                }
            updated_tracks.append(det)

        now = time.time()
        self.tracks = {
            tid: trk for tid, trk in self.tracks.items()
            if now - trk["last_seen"] <= self.max_age
        }
        return updated_tracks

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
