from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torch
import easyocr
import re

RUS_PLATE_REGEX = re.compile(r'^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$')
CONF_THRESHOLD = 0.7
last_plate_bbox = None
last_plate_text = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

car_model = YOLO('models/cars.pt')
plate_detector = YOLO('models/plates.pt')
car_model.to(device)
plate_detector.to(device)

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
tracker = DeepSort(max_age=60, n_init=2, max_iou_distance=0.7, nms_max_overlap=1.0)
cap = cv2.VideoCapture('cars.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = car_model(frame, conf=0.5, iou=0.45)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = car_model.names[cls_id]
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)
    best_track = None
    max_area = 0
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_track = track

    if best_track:
        x1, y1, x2, y2 = map(int, best_track.to_ltrb())
        car_roi = frame[y1:y2, x1:x2]
        if car_roi.size and car_roi.shape[0] > 0 and car_roi.shape[1] > 0:
            plate_results = plate_detector(car_roi, conf=0.3)[0]
            if plate_results.boxes:
                bx1, by1, bx2, by2 = map(int, plate_results.boxes.xyxy[0])
                plate_conf = float(plate_results.boxes.conf[0])
                full_bbox = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)

                if plate_conf >= CONF_THRESHOLD:
                    plate_crop = frame[full_bbox[1]:full_bbox[3], full_bbox[0]:full_bbox[2]]
                    ocr_results = reader.readtext(plate_crop)
                    if ocr_results:
                        text = re.sub(r'[^A-Z0-9]', '', ocr_results[0][1].upper())
                        if RUS_PLATE_REGEX.match(text):
                            last_plate_bbox = full_bbox
                            last_plate_text = text

    if last_plate_bbox and not last_plate_text:
        x1, y1, x2, y2 = last_plate_bbox
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size and plate_crop.shape[0] > 0 and plate_crop.shape[1] > 0:
            ocr_results = reader.readtext(plate_crop)
            if ocr_results:
                text = re.sub(r'[^A-Z0-9]', '', ocr_results[0][1].upper())
                if RUS_PLATE_REGEX.match(text):
                    last_plate_text = text

    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
        label = f"ID:{tid}"
        if last_plate_text and best_track and track.track_id == best_track.track_id:
            label += f" {last_plate_text}"
        cv2.putText(frame, label, (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()