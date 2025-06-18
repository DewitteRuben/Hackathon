import cv2
import supervision as sv
from ultralytics import YOLO
import socketio
import base64
from datetime import datetime
import time

model = YOLO("yolov8n_ncnn_model")
label_annotator = sv.LabelAnnotator()
sio = socketio.Client()

box_annotator = sv.BoxAnnotator()

@sio.event
def connect():
    print('Connected to server')

@sio.event
def disconnect():
    print('Disconnected from server')

try:
    sio.connect('http://192.168.178.135:3000', wait_timeout=10)
except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)

while True:
    # Capture new picture each time
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()  # Release the camera immediately after capturing
    
    if not ret:
        print("Failed to capture image")
        continue

    # Run inference with optimized settings
    result = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    # Count bananas and get their centroids
    count = 0
    centroids = []
    if detections.class_id is not None:
        for i, class_id in enumerate(detections.class_id):
            if model.names[class_id] == 'banana':
                count += 1
                box = detections.xyxy[i]
                x1, y1, x2, y2 = box
                centroid = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                centroids.append(centroid)
    
    if detections.class_id is not None and detections.confidence is not None:
        labels = [f"{model.names[class_id]} {confidence:0.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    else:
        labels = []
    
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    retval, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpg_as_text = base64.b64encode(buffer.tobytes())
    
    message = {
        'count': count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'centroid': centroids,
    }
    
    try:
        sio.emit('message', message)
    except Exception as e:
        print(f"Error sending message: {e}")
        cv2.destroyAllWindows()
        sio.disconnect()