import os
import cv2

def detect_and_save_violations(input_path, mode="image"):
    os.makedirs("outputs", exist_ok=True)

    # Dummy detection (replace with your actual logic)
    output_file = f"outputs/processed_{os.path.basename(input_path)}"
    cv2.imwrite(output_file, cv2.imread(input_path))

    return [output_file]
