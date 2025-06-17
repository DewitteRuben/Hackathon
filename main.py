import cv2
import supervision as sv
from ultralytics import YOLO
import socketio
import base64

model = YOLO("yolov8s.pt")
label_annotator = sv.LabelAnnotator()
sio = socketio.SimpleClient()

box_annotator = sv.BoxAnnotator()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    count = len(detections)
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    # Save the annotated frame to a file
    retval, buffer = cv2.imencode('.jpg', annotated_frame)
    jpg_as_text = base64.b64encode(buffer.tobytes())

    # sio.emit('image', jpg_as_text)
    # sio.emit('count', count)
    # print(jpg_as_text)
    print(count)
    break

cap.release()
cv2.destroyAllWindows()