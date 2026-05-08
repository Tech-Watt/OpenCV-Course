# OpenCV Course

This repository contains beginner-friendly OpenCV examples for learning image processing, video processing, drawing tools, image transformations, edge detection, and object detection with YOLO.

The course code is organized inside `main.py` as separate commented modules. You can uncomment the section you want to practice and run it step by step.

## Project Structure

| File | Description |
| --- | --- |
| `main.py` | Main course script containing OpenCV examples and YOLO object detection code. |
| `car.jpg` | Sample image used for loading, resizing, and grayscale examples. |
| `soccer.jpg` | Sample image used for drawing, transformations, blur, and edge detection. |
| `dance1.mp4` | Sample video used for video reading/writing examples. |
| `dance2.mp4` | Additional sample video asset. |
| `output.mp4` | Example processed video output. |
| `ball.jpg`, `line.png`, `rect.png`, `rectangle.png`, `object detection.jpg`, `codecs.PNG` | Supporting image assets for course examples. |
| `yolo11n.pt` | YOLO model checkpoint included in the repository. |
| `runs/` | Output folder created by YOLO/Ultralytics runs. |

## Topics Covered

- Loading and displaying images
- Resizing images
- Converting images to grayscale
- Drawing shapes, text, lines, and arrows
- Reading and writing video files
- Cropping and rotating images
- Blurring images
- Canny edge detection
- Object detection with Ultralytics YOLO
- Reading bounding boxes, class names, and confidence scores

## Requirements

- Python 3.8 or newer
- OpenCV
- Ultralytics
- A webcam if you want to run live object detection

Install dependencies:

```bash
pip install opencv-python ultralytics
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Tech-Watt/OpenCV-Course.git
cd OpenCV-Course
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install opencv-python ultralytics
```

## How to Use the Course File

Open `main.py` and choose the module you want to run.

Most modules are commented out so you can focus on one topic at a time. Uncomment one section, run it, then comment it again before trying another section.

Run the script:

```bash
python main.py
```

For image examples, OpenCV will open a display window. Press any key or close the window depending on the example.

For video examples, press `q` when the video window is active to stop playback.

## Course Modules

### Module 1: Image Basics

Examples include:

- `cv2.imread`
- `cv2.imshow`
- `cv2.resize`
- `cv2.cvtColor`
- `cv2.waitKey`
- `cv2.destroyAllWindows`

This module uses `car.jpg`.

### Module 2: Drawing on Images

Examples include:

- Circles
- Rectangles
- Lines
- Text
- Ellipses
- Arrowed lines

This module uses `soccer.jpg`.

### Module 3: Working with Video

Examples include:

- Reading video with `cv2.VideoCapture`
- Getting video width and height
- Resizing video frames
- Writing output video with `cv2.VideoWriter`
- Looping through frames

This module uses `dance1.mp4`.

### Module 4: Image Transformations

Examples include:

- Cropping an image
- Rotating an image
- Creating a rotation matrix
- Applying affine transforms

This module uses `soccer.jpg`.

### Module 5: Blur and Edge Detection

Examples include:

- Gaussian blur
- Average blur
- Canny edge detection

This module uses `soccer.jpg`.

### Module 6: Object Detection

The active section of `main.py` uses Ultralytics YOLO for object detection:

```python
from ultralytics import YOLO

model = YOLO('yolo12.pt')
results = model(source=1, show=True)
```

The repository includes `yolo11n.pt`, so update the model path if needed:

```python
model = YOLO('yolo11n.pt')
```

The code prints detected class names, confidence scores, and bounding box coordinates:

```python
names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
confs = result.boxes.conf
xywh = result.boxes.xywh
```

## Notes

- Keep only one course section active at a time for easier debugging.
- If `cv2.imshow` does not work in your environment, run the script locally instead of inside a headless notebook/server.
- If live object detection does not start, check your webcam index in `source=1`. Many systems use `source=0`.
- Large videos and YOLO output folders can make the repository large. For new experiments, consider ignoring generated output files.
- The included course assets are meant for practice and demonstration.

## Author

Created by [Tech Watt](https://github.com/Tech-Watt).
