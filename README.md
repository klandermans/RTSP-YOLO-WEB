# Multi-Camera YOLOv8 Livestream with Keypoints to Website

This demo was developed as part of the **Next Level Animal Science** knowledge-based program at Wageningen University & Research. The project aims to study dairy cow behavior using real-time video analysis with keypoints. While the YOLOv8-based tracking model was trained by research collaborators, this application showcases how to connect multiple IP cameras, overlay keypoint visualizations, and stream the output directly to a browser — even from within a protected network.

Unlike many applications that analyze pre-recorded MP4 videos, this project is built entirely around **real-time inference**. Every frame is processed live by a YOLOv8 model using a multi-threaded setup that enables response times within approximately 20 seconds. This makes it suitable for real-time monitoring and behavior analysis without storing video data.

![image](https://github.com/user-attachments/assets/9eb425fb-9811-49d6-8eee-ee640f544211)


## What It Does

- Connects to multiple RTSP IP camera feeds
- Performs real-time object tracking using [YOLOv8](https://github.com/ultralytics/ultralytics)
- Visualizes bounding boxes, object IDs, and keypoints with custom styling
- Streams each camera feed live to a web interface via Flask
- Displays current camera status (loading, ok, offline)

## Features

- Multi-threaded real-time inference
- Keypoint overlays and color-coded IDs
- 2D line connections between keypoints
- Lightweight Flask-based viewer

## Example RTSP Setup

```python
CAMERA_URLS = {
    "cam1": "rtsp://<username>:<password>@10.82.16.217//onvif-media/media.amp",
    "cam7": "rtsp://<username>:<password>@10.82.16.224//onvif-media/media.amp",
    "cam2": "rtsp://<username>:<password>@10.82.16.218//onvif-media/media.amp",
    "cam4": "rtsp://<username>:<password>@10.82.16.221//onvif-media/media.amp",
}
```

> Do **not** commit real passwords or internal IP addresses to version control.
> Use environment variables or a config file excluded via `.gitignore`.

## ⚙️ Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV (`opencv-python`)
- Flask
- NumPy
- FFmpeg (optional for YouTube streaming)

## Installation

```bash
pip install ultralytics opencv-python numpy flask
```

## Running

```bash
python multi_camera_stream.py
```

Then open your browser at: [http://localhost:9999](http://localhost:9999)

---

Feel free to extend this project for custom layouts, external streaming (e.g. YouTube), or additional analytics like activity heatmaps or behavioral classification.
