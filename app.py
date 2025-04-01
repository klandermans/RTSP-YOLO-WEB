# multi_camera_stream.py
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import cv2
import threading
import time
import numpy as np

app = Flask(__name__)

# Define your RTSP URLs here (replace with secrets or environment variables in production)
CAMERA_URLS = {
    "cam1": "rtsp://<username>:<password>@10.82.16.217//onvif-media/media.amp",
    "cam7": "rtsp://<username>:<password>@10.82.16.224//onvif-media/media.amp",
    "cam2": "rtsp://<username>:<password>@10.82.16.218//onvif-media/media.amp",
    "cam4": "rtsp://<username>:<password>@10.82.16.221//onvif-media/media.amp",
}

camera_status = {cam_id: 'loading' for cam_id in CAMERA_URLS}
models = {cam_id: YOLO("models/yolov8/best.pt") for cam_id in CAMERA_URLS}
frame_lock = threading.Lock()
latest_frames = {cam_id: None for cam_id in CAMERA_URLS}
id_to_color = {}

CAMERA_ID_OFFSET = {
    "cam1": 0,
    "cam2": 1000,
    "cam4": 2000,
    "cam7": 3000,
}

CUSTOM_CONNECTIONS = [(0, 2), (1, 2), (2, 3)]

def get_color_from_id(id_val):
    if id_val not in id_to_color:
        np.random.seed(int(id_val))
        id_to_color[id_val] = tuple(np.random.randint(0, 255, size=3).tolist())
    return id_to_color[id_val]

def draw_img_results(img, boxes, keypoints, ids=None, font_scale=0.9):
    try:
        if keypoints is None or len(keypoints) == 0:
            return img
    except TypeError:
        return img

    for i, kps in enumerate(keypoints):
        if len(kps) < 4:
            continue

        color = get_color_from_id(ids[i]) if ids is not None else (0, 255, 0)

        for p1, p2 in CUSTOM_CONNECTIONS:
            x1, y1 = int(kps[p1][0]), int(kps[p1][1])
            x2, y2 = int(kps[p2][0]), int(kps[p2][1])
            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                cv2.line(img, (x1, y1), (x2, y2), color, 2)

        for kp in kps[:4]:
            x, y = int(kp[0]), int(kp[1])
            if (x, y) != (0, 0):
                cv2.circle(img, (x, y), 10, color, -1)

        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cow_number = ids[i] if ids is not None else i + 1
            cv2.putText(img, f"Cow {cow_number}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return img

def process_camera(cam_id, rtsp_url):
    global camera_status
    try:
        results = models[cam_id].track(source=rtsp_url, stream=True, conf=0.5, iou=0.5, tracker="bytetrack.yaml")
        for result in results:
            try:
                if result is None:
                    continue

                frame = result.orig_img
                bboxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
                ids = result.boxes.id.cpu().numpy() if result.boxes and result.boxes.id is not None else []
                if len(ids) > 0:
                    ids = ids + CAMERA_ID_OFFSET.get(cam_id, 0)

                keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints and result.keypoints.xy is not None else []
                frame = draw_img_results(frame, bboxes, keypoints=keypoints, ids=ids)

                with frame_lock:
                    latest_frames[cam_id] = frame

                camera_status[cam_id] = 'ok'
            except Exception as e:
                print(f"[{cam_id}] Prediction error in stream loop: {e}")
                camera_status[cam_id] = 'offline'
                time.sleep(2)
    except Exception as e:
        print(f"[{cam_id}] Error starting YOLO tracking stream: {e}")
        camera_status[cam_id] = 'offline'
        time.sleep(3)
        process_camera(cam_id, rtsp_url)

@app.route('/')
def index():
    html = ""
    for cam_id in CAMERA_URLS:
        html += f'''<img src="/video_feed/{cam_id}" width="45%">'''
    return render_template_string(html)

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    def generate():
        while True:
            with frame_lock:
                frame = latest_frames.get(cam_id)

            if frame is not None:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_bytes = buffer.tobytes()
            else:
                status = camera_status.get(cam_id, 'loading')
                placeholder = np.zeros((320, 640, 3), dtype=np.uint8)
                color = (0, 255, 0) if status == 'ok' else (0, 255, 255) if status == 'loading' else (0, 0, 255)
                cv2.putText(placeholder, f"Camera {cam_id}: {status.upper()}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.2)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    for cam_id, url in CAMERA_URLS.items():
        t = threading.Thread(target=process_camera, args=(cam_id, url), daemon=True)
        t.start()
    app.run(host='0.0.0.0', port=9999)
