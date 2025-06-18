import cv2
import supervision as sv
from ultralytics import YOLO
import socketio
import base64
from datetime import datetime, timezone
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

cap = cv2.VideoCapture(0)

while True:
    # Capture new picture each time
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        continue

    # Run inference with optimized settings
    result = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
        
    if detections.class_id is not None:
        # Check if 'banana' exists in model names
        if 'banana' in model.names.values():
            banana_class_id = list(model.names.keys())[list(model.names.values()).index('banana')]
            banana_mask = detections.class_id == banana_class_id
            detections = detections[banana_mask]
        else:
            print("Warning: 'banana' class not found in model")
    
    if detections.class_id is not None and detections.confidence is not None:
        labels = [f"{model.names[class_id]} {confidence:0.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    else:
        labels = []
    
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    retval, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpg_as_text = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    message = {
        'image': jpg_as_text,
        'count': len(detections),
        'timestamp': timestamp,
    }
    
    try:
        sio.emit('message', message)
    except Exception as e:
        print(f"Error sending message: {e}")
        cv2.destroyAllWindows()
        cap.release()
        sio.disconnect()