import cv2
from ultralytics import YOLO

# Load your YOLO model (the one already working)
model = YOLO("yolov8n.pt")   # You can replace with your custom model later

def detect_violations(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video!")
        return 0, 0, 0, 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # counters
    red_light_violations = set()
    helmet_violations = set()
    wrong_side_violations = set()

    stop_line_y = int(height * 0.55)

    # NEW → detect if ANY traffic light exists in video
    red_light_present = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect using YOLO
        results = model(frame, verbose=False)

        for det in results:
            for box in det.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ----------------------------------
                # 1. Traffic light detection
                # ----------------------------------
                if label in ["traffic light", "traffic_light", "light"] and conf > 0.50:
                    # YOLO cannot tell red/green
                    # So just check if a traffic light exists at all
                    red_light_present = True

                # ----------------------------------
                # 2. Red light violation (only if red light exists)
                # ----------------------------------
                if red_light_present:
                    if label in ["car", "motorbike", "bus", "truck"]:
                        if y2 > stop_line_y:
                            red_light_violations.add((x1, y1))

                # ----------------------------------
                # 3. Helmet detection placeholder
                # ----------------------------------
                if label == "motorbike":
                    helmet_violations.add((x1, y1))

                # ----------------------------------
                # 4. Wrong-side detection placeholder
                # ----------------------------------
                wrong_side_violations.add((x1, y1))

        out.write(frame)

    cap.release()
    out.release()

    # ----------------------------------
    # IMPORTANT FIX:
    # If no traffic lights were found,
    # → no red light violations possible
    # ----------------------------------
    if not red_light_present:
        red_light_violations = set()

    total = len(red_light_violations) + len(helmet_violations) + len(wrong_side_violations)

    return (
        len(red_light_violations),
        len(helmet_violations),
        len(wrong_side_violations),
        total
    )
