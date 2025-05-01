from filterpy.kalman import KalmanFilter
import numpy as np

class Sort:
    def __init__(self, max_age=5, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.track_id = 0

    def update(self, detections):
        # Step 1: Increment age of all existing trackers
        for tracker in self.trackers:
            tracker.increment_age()

        updated_trackers = []
        assigned_trackers = set()

        # Step 2: Match detections to existing trackers using IOU
        for det in detections:
            assigned = False
            for tracker in self.trackers:
                if tracker in assigned_trackers:
                    continue
                if self._iou(tracker.bbox, det['bbox']) > 0.3:
                    tracker.update(det['bbox'])
                    tracker.age = 0  # reset age if updated
                    updated_trackers.append(tracker)
                    assigned_trackers.add(tracker)
                    assigned = True
                    break
            if not assigned:
                new_tracker = Tracker(self.track_id, det['bbox'])
                self.track_id += 1
                updated_trackers.append(new_tracker)

        # Step 3: Keep unassigned trackers if not too old
        for tracker in self.trackers:
            if tracker not in assigned_trackers and tracker.age < self.max_age:
                updated_trackers.append(tracker)

        self.trackers = updated_trackers

        return [{"bbox": tracker.bbox, "track_id": tracker.track_id} for tracker in self.trackers]

    def _iou(self, boxA, boxB):
        x1, y1, x2, y2 = boxA
        xa, ya, xb, yb = boxB

        # compute the intersection area
        inter_area = max(0, min(x2, xb) - max(x1, xa)) * max(0, min(y2, yb) - max(y1, ya))

        # compute the areas
        areaA = (x2 - x1) * (y2 - y1)
        areaB = (xb - xa) * (yb - ya)

        # compute the union area
        union_area = areaA + areaB - inter_area

        return inter_area / union_area if union_area != 0 else 0

class Tracker:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.age = 0  # How many frames the tracker has existed
    
    def update(self, bbox):
        self.bbox = bbox
        self.age = 0
    
    def increment_age(self):
        self.age += 1
