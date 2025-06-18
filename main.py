import cv2
import supervision as sv
from ultralytics import YOLO
import socketio
import base64
from datetime import datetime, timezone
import time
from gpiozero import LED, Button
from signal import pause

led = LED(17) # GPIO17 (physical pin 11)
button = Button(26) # GPIO26 (physical pin 37)

# Flag to control label display
show_labels = True

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")
    return cap

def reconnect_camera():
    global cap
    print("Attempting to reconnect camera...")
    if cap is not None:
        cap.release()
    time.sleep(1)  # Wait a bit before reconnecting
    cap = initialize_camera()
    print("Camera reconnected successfully")

def reconnect_socket():
    global sio
    print("Attempting to reconnect socket...")
    try:
        # First check if we're actually disconnected
        if not sio.connected:
            # If we're not connected, try to connect
            sio.connect('http://192.168.178.135:3000', wait_timeout=10)
            print("Socket reconnected successfully")
            return True
        else:
            # If we're already connected, try to emit a test message
            try:
                sio.emit('test_connection', {'test': True})
                print("Socket is already connected and working")
                return True
            except Exception as e:
                # If test message fails, force disconnect and reconnect
                print("Socket appears connected but test failed, forcing reconnection...")
                sio.disconnect()
                time.sleep(1)  # Wait a bit before reconnecting
                sio.connect('http://192.168.178.135:3000', wait_timeout=10)
                print("Socket reconnected successfully")
                return True
    except Exception as e:
        print(f"Socket reconnection failed: {e}")
        return False

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

# Initialize camera
cap = initialize_camera()
reconnect_attempts = 0
max_reconnect_attempts = 10000
socket_reconnect_attempts = 0
max_socket_reconnect_attempts = 10000

while True:
    # Capture new picture each time
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        if reconnect_attempts < max_reconnect_attempts:
            reconnect_attempts += 1
            reconnect_camera()
            continue
        else:
            print("Max reconnection attempts reached. Exiting...")
            break

    # Reset reconnect attempts on successful capture
    reconnect_attempts = 0

    result = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
        
    if detections.class_id is not None:
        # Check if 'banana' and 'bird' exist in model names
        target_classes = ['banana', 'bird']
        valid_detections = []
        
        for class_name in target_classes:
            if class_name in model.names.values():
                class_id = list(model.names.keys())[list(model.names.values()).index(class_name)]
                class_mask = detections.class_id == class_id
                valid_detections.append(detections[class_mask])
            else:
                print(f"Warning: '{class_name}' class not found in model")
        
        if valid_detections:
            # Combine all valid detections
            detections = sv.Detections.merge(valid_detections)
            
            # Filter out low confidence detections
            confidence_threshold = 0.4
            confidence_mask = detections.confidence >= confidence_threshold
            detections = detections[confidence_mask]
    
    if detections.class_id is not None and detections.confidence is not None:
        labels = [f"{model.names[class_id]} {confidence:0.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    else:
        labels = []
    
    def button_pressed():
        global show_labels
        show_labels = not show_labels
        print(f"Button pressed! Labels display: {show_labels}")

    button.when_pressed = button_pressed
    
    actual_frame = frame.copy()
    if show_labels:
        led.on()
        actual_frame = box_annotator.annotate(scene=actual_frame, detections=detections)
        actual_frame = label_annotator.annotate(scene=actual_frame, detections=detections, labels=labels)
    else:
        led.off()

    retval, buffer = cv2.imencode('.jpg', actual_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    jpg_as_text = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    message = {
        'image': jpg_as_text,
        'count': len(detections),
        'timestamp': timestamp,
    }
    
    try:
        sio.emit('message', message)
        time.sleep(0.5)
        # Reset socket reconnect attempts on successful send
        socket_reconnect_attempts = 0
    except Exception as e:
        print(f"Error sending message: {e}")
        if socket_reconnect_attempts < max_socket_reconnect_attempts:
            socket_reconnect_attempts += 1
            if reconnect_socket():
                # After successful reconnection, continue with the next frame
                continue
            else:
                print(f"Socket reconnection attempt {socket_reconnect_attempts} failed")
                # Even if reconnection fails, continue with the next frame
                continue
        else:
            print("Max socket reconnection attempts reached. Exiting...")
            cv2.destroyAllWindows()
            cap.release()
            sio.disconnect()
            break