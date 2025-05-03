# Edge-Lite Object Tracker

This project is a real-time object detection and tracking system using the YOLOv5 model and the SORT (Simple Online and Realtime Tracking) algorithm. It captures video from a webcam, detects objects, tracks them, and outputs both a video and individual frames with bounding boxes and tracking IDs.

## Features

* Real-time object detection using a pre-trained YOLOv5 model.
* Object tracking with the SORT algorithm.
* Frame saving and video recording with bounding boxes.
* Object lifecycle tracking, including start and end times.
* Output video and individual frame images stored in a folder.

## Prerequisites

* Python 3.6 or higher
* OpenCV
* ONNX Runtime
* NumPy
* Other dependencies (listed in requirements)

## Setup and Installation

### Step 1: Clone the repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/edge-lite-object-tracker.git
cd edge-lite-object-tracker
```

### Step 2: Install dependencies

You can install the required dependencies using pip. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

Here is a sample `requirements.txt`:

```
opencv-python
onnxruntime
numpy
```

### Step 3: Download YOLOv5 Model

Download the YOLOv5 ONNX model (`yolov5s.onnx`) from [the official YOLOv5 repository](https://github.com/ultralytics/yolov5/releases) or another reliable source and place it in the `models/` directory.

### Step 4: Run the Program

Now you can run the program to start the object detection and tracking:

```bash
python src/main.py
```

This will start capturing video from your webcam, detect objects, and track them. The processed frames will be saved in the `output_frames/` folder, and the video will be saved as `output_video.mp4`.

You can press `'q'` to stop the process.

## How It Works

1. **Detection**:

   * The system uses a YOLOv5 model (converted to ONNX format) for object detection. The model detects objects and returns their bounding boxes and labels.

2. **Tracking**:

   * The SORT algorithm is used to track the detected objects across multiple frames. Each object is assigned a unique tracking ID.
   * The system performs tracking by comparing the detected bounding boxes using the Intersection over Union (IoU) metric to associate objects across frames.

3. **Output**:

   * For each detected object, bounding boxes and tracking IDs are drawn on the frames.
   * The processed frames are saved as individual images in the `output_frames/` folder.
   * The processed video is saved as `output_video.mp4`.

4. **Lifecycle Tracking**:

   * Each object’s lifecycle, including the start and end times, is tracked and stored in a CSV file (`tracking_data.csv`).

## Project Structure

```
edge-lite-object-tracker/
├── models/                  # Store your YOLOv5 model here (e.g., yolov5s.onnx)
├── output_frames/           # Folder where the processed frames will be saved
├── src/
│   ├── main.py              # Main script for running the program
│   ├── detector.py          # Object detection using YOLOv5
│   └── tracker.py           # Tracking algorithm (SORT)
└── requirements.txt         # Python dependencies
```

## File Descriptions

* **main.py**:

  * Contains the main function to run the video capture, object detection, and tracking.
  * Initializes the `ObjectDetector` and `Sort` tracker and handles video output.

* **detector.py**:

  * Defines the `ObjectDetector` class that loads and uses the YOLOv5 model to detect objects.
  * Includes helper functions like non-maximum suppression (NMS) for filtering overlapping bounding boxes.

* **tracker.py**:

  * Implements the SORT tracking algorithm.
  * Tracks the objects using the bounding boxes across frames and assigns unique tracking IDs.

## Configuration

You can adjust the following parameters to suit your needs:

* **max\_age**: The maximum number of frames that an object can be unseen before it is removed from tracking.
* **min\_hits**: The minimum number of frames an object must be seen to start tracking.
* **iou\_threshold**: The Intersection over Union (IoU) threshold used for matching detected objects to tracked objects. You can adjust this threshold in the `Sort` class constructor if required.

## Output

1. **Frames**:

   * All frames with object bounding boxes will be saved as individual images in the `output_frames/` folder.

2. **Video**:

   * A video file named `output_video.mp4` will be saved with the tracked objects.

3. **CSV**:

   * The `tracking_data.csv` file contains the following columns:

     * `track_id`: The unique ID assigned to each tracked object.
     * `label`: The object label (e.g., "person", "car").
     * `x1`, `y1`, `x2`, `y2`: Coordinates of the bounding box.
     * `start_time`: The timestamp when the object was first detected.
     * `end_time`: The timestamp when the object was last seen.

