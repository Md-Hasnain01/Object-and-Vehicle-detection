from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO('yolov8l.pt')

className = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", " dog",
             "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
             "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", " sandwich", "orange",
             "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
             "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
             ]

cap = cv2.VideoCapture("C:\\Users\\mh183\\PycharmProjects\\traffic\\yolo\\360_F_458083018_GWf89ndA62i30YOJBD7rjUd30uXylPO5.jpg")
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            # class name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)

    cv2.imshow("Image", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()