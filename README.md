Sure! Here's the full **README.md** content in a single Markdown-formatted block that you can copy and paste directly into your project as `README.md`:

---

```markdown
# Edge-Lite Object Tracker

**Edge-Lite Object Tracker** is a lightweight real-time object detection and tracking system. It uses a YOLOv5 model (in ONNX format) for object detection and a simple SORT (Simple Online and Realtime Tracking) algorithm for tracking. Designed for efficiency, it's suitable for edge devices like Raspberry Pi, Jetson Nano, and lightweight desktops.

---

## 🚀 Features

- ✅ Real-time object detection with YOLOv5 (ONNX)
- ✅ Custom SORT tracking implementation
- ✅ OpenCV GUI with live FPS display
- ✅ Frame skipping for better performance
- ✅ Clean, modular codebase (easily extendable)
- ✅ Close window with GUI button or `q` key

---

## 📁 Project Structure

```
edge-lite-object-tracker/
├── models/                # ONNX models (e.g. yolov5n.onnx)
├── src/
│   ├── main.py            # Main execution script
│   ├── detector.py        # Object detection logic
│   ├── tracker.py         # Object tracking logic
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🧩 Prerequisites

Make sure you have Python 3.7+ installed.

Install required packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python numpy onnxruntime filterpy
```

---

## ▶️ Running the Tracker

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/edge-lite-object-tracker.git
cd edge-lite-object-tracker
```

2. **Put your model in the `models/` directory:**

> e.g. `models/yolov5n.onnx`  
> You can download YOLOv5 models and convert them to ONNX using [Ultralytics](https://github.com/ultralytics/yolov5).

3. **Run the main script:**

```bash
python src/main.py
```

- Press `q` to quit
- Or use the Close button in the top-left of the OpenCV window

---

## 🎥 Use a Video File Instead of Webcam

In `src/main.py`, change this line:

```python
cap = cv2.VideoCapture(0)
```

to:

```python
cap = cv2.VideoCapture("path/to/video.mp4")
```

---

## ⚙️ Tuning for Better Results

- **Too many boxes?**  
  Increase `conf_threshold` in `detector.py`:

```python
def detect(self, frame, conf_threshold=0.3):
```

- **Track stability issues?**  
  Adjust the IOU threshold in `tracker.py`:

```python
if self._iou(tracker.bbox, det['bbox']) > 0.3:
```

---

## 🛠 Close Button in OpenCV

You can click the `X` or top-left corner of the OpenCV window to close the app gracefully. It uses `cv2.destroyAllWindows()` at the end of `main.py`.

---

## 📄 License

MIT License © 2025  
Free to use and modify.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [SORT Tracking Algorithm](https://github.com/abewley/sort)
- [OpenCV](https://opencv.org/)
```