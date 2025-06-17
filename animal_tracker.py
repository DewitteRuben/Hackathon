import cv2
import supervision as sv
from ultralytics import YOLO
import os
from datetime import datetime
import time

# Create directory for saving images if it doesn't exist
SAVE_DIR = "detected_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# List of target animals to track
TARGET_ANIMALS = [
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
]

model = YOLO("yolov8s.pt")
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

while True:
    # Capture new picture
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()  # Release the camera immediately after capturing
    
    if not ret:
        print("Failed to capture image")
        time.sleep(2)
        continue

    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    
    # Check if any objects were detected
    if detections.class_id is not None and len(detections.class_id) > 0:
        # Get labels for all detected objects
        labels = [f"{model.names[class_id]} {confidence:0.3f}" 
                 for class_id, confidence in zip(detections.class_id, detections.confidence)]
        
        # Check if any of the detected objects are in our target animals list
        detected_animals = [label.split()[0] for label in labels]
        if any(animal in TARGET_ANIMALS for animal in detected_animals):
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save the original image
            original_filename = f"{SAVE_DIR}/original_{timestamp}.jpg"
            cv2.imwrite(original_filename, frame)
            
            # Annotate the frame
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            # Save the annotated image
            annotated_filename = f"{SAVE_DIR}/annotated_{timestamp}.jpg"
            cv2.imwrite(annotated_filename, annotated_frame)
            
            print(f"Saved images: {original_filename} and {annotated_filename}")
            print(f"Detected animals: {', '.join(labels)}")
    
    # Wait for 0.5 second before next iteration
    time.sleep(0.5)

cv2.destroyAllWindows()
