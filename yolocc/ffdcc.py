import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

model = YOLO('yolov8l.pt')

className = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "traffic light",
             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", " dog",
             "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
             "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
             "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
             "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", " sandwich", "orange",
             "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
             "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
             "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
             "teddy bear", "hair drier", "toothbrush"
             ]

cap = cv2.VideoCapture("C:\\Users\\mh183\\PycharmProjects\\traffic\\yolocc\\WhatsApp Video 2024-09-08 at 19.41.54_013fe5cb.mp4")
# cap.set(3, 1280)
# cap.set(4, 720)

# tracking
tracker = Sort(max_age=20, min_hits=2)

# Define ROI for the traffic light area
roi_start_point = (300, 500)  # Top-left corner of the ROI
roi_end_point = (1000, 700)   # Bottom-right corner of the ROI

# Initialize a dictionary to store vehicle positions
vehicle_positions = {}

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # class name
            cls = int(box.cls[0])
            currentClass = className[cls]
            if currentClass == "car" or currentClass == "motorbike" or currentClass == "bus" or currentClass == "truck":
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)
                cvzone.cornerRect(img, (x1, y1, w, h))
                currentArray = np.array([x1, x2, y1, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    stopped_vehicle_count = 0

    for result in resultsTracker:
        x1, x2, y1, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        # Check if the vehicle is within the ROI
        if roi_start_point[0] <= x1 <= roi_end_point[0] and roi_start_point[1] <= y1 <= roi_end_point[1]:
            if Id not in vehicle_positions:
                vehicle_positions[Id] = (x1, y1)
            else:
                # Calculate movement by comparing the current position with the previous one
                prev_x1, prev_y1 = vehicle_positions[Id]
                movement = math.sqrt((x1 - prev_x1) ** 2 + (y1 - prev_y1) ** 2)

                # If the movement is below a threshold, consider the vehicle as stopped
                if movement < 2.0:
                    stopped_vehicle_count += 1

                vehicle_positions[Id] = (x1, y1)

        cvzone.cornerRect(img, (x1, y1, w, h))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=2)

        # Draw the ROI on the image
        cv2.rectangle(img, roi_start_point, roi_end_point, (0, 255, 0), 2)
        cv2.putText(img, f'Stopped Vehicles: {stopped_vehicle_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
