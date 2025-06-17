import cv2
import supervision as sv
from ultralytics import YOLO
import socketio
import base64
from datetime import datetime
import time

model = YOLO("yolov8s.pt")
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
        time.sleep(20)
        continue

    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    # Count only bananas
    count = 0
    if detections.class_id is not None:
        for class_id in detections.class_id:
            if model.names[class_id] == 'banana':
                count += 1
    
    if detections.class_id is not None and detections.confidence is not None:
        labels = [f"{model.names[class_id]} {confidence:0.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    else:
        labels = []
    
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    # Save the annotated frame to a file
    retval, buffer = cv2.imencode('.jpg', annotated_frame)
    jpg_as_text = base64.b64encode(buffer.tobytes())
    
    x1, y1, x2, y2 = box 
    centroid = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

    # Create message with banana count and timestamp
    message = {
        'count': count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'centroid': centroid,
    }
    
    print(f"Sending: {message}")
    try:
        sio.emit('message', message)
        print("Message sent successfully")
    except Exception as e:
        print(f"Error sending message: {e}")
    
    # Wait for 20 seconds before next iteration
    print("Waiting 20 seconds before next detection...")
    time.sleep(20)

cv2.destroyAllWindows()
sio.disconnect()