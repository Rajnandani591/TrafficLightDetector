import cv2
import pytesseract
import os
from datetime import datetime
from ultralytics import YOLO

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO("yolov8n.pt")  # replace with your YOLO model path

seen_plates = set()

def boxes_overlap(box1, box2, threshold=0.3):
    """
    Returns True if box1 and box2 overlap more than threshold.
    box = (x1, y1, x2, y2)
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou > threshold

def draw_violations(frame, results):
    """
    Draws bounding boxes for motorcycles with rider and highlights violations.
    """
    global seen_plates
    violations = []

    motorcycles = []
    persons = []

    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls = int(classes[i])

            if cls == 3:  # motorcycle
                motorcycles.append((x1, y1, x2, y2))
            elif cls == 0:  # person
                persons.append((x1, y1, x2, y2))

    for moto_box in motorcycles:
        x1, y1, x2, y2 = moto_box

        # Check if a person is riding
        rider_present = False
        for person_box in persons:
            if boxes_overlap(moto_box, person_box, threshold=0.2):
                rider_present = True
                break

        if not rider_present:
            continue  # skip parked bikes

        # Crop for plate OCR
        cropped = frame[y1:y2, x1:x2]
        plate_text = ""
        try:
            plate_text = pytesseract.image_to_string(cropped, config='--psm 7').strip()
        except Exception:
            pass

        if plate_text and plate_text in seen_plates:
            continue
        if plate_text:
            seen_plates.add(plate_text)

        # Set violation rule - for now, helmet as placeholder
        rule_name = "Helmet Violation"
        color = (0, 0, 255)  # bright red for visibility

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw text inside the box
        text = f"{rule_name} | Plate: {plate_text}" if plate_text else rule_name
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
        cv2.putText(frame, text, (x1, y1 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        violations.append({'rule': rule_name, 'plate': plate_text, 'time': timestamp})

    return frame, violations

def process_file(path, result_folder):
    """
    Processes image or video. Saves only frames with active 2-wheeler violations.
    1 frame every 2 seconds for video.
    """
    ext = os.path.splitext(path)[1].lower()
    result_files = []
    violations_info = []

    if ext in ['.jpg', '.png', '.jpeg']:
        img = cv2.imread(path)
        img = cv2.resize(img, (640, 480))
        results = model.predict(img)
        frame, violations = draw_violations(img, results)
        out_file = os.path.join(result_folder, os.path.basename(path))
        cv2.imwrite(out_file, frame)
        result_files.append(out_file)
        violations_info.extend(violations)

    else:
        cap = cv2.VideoCapture(path)
        video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        skip_frames = max(1, video_fps * 2)  # 1 frame per 2 seconds
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                frame = cv2.resize(frame, (640, 480))
                results = model.predict(frame)
                frame, violations = draw_violations(frame, results)

                if violations:
                    out_file = os.path.join(result_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(out_file, frame)
                    result_files.append(out_file)
                    violations_info.extend(violations)

            frame_count += 1

        cap.release()

    return result_files, violations_info
