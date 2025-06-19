import cv2
import numpy as np
from picamera2 import Picamera2
import libcamera
import tflite_runtime.interpreter as tflite

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(image, M, (nW, nH))

# === 1. Load TFLite model ===
interpreter = tflite.Interpreter(model_path="model/model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # [1, 128, 128, 3]
height, width = input_shape[1], input_shape[2]

# === 2. Initialize camera ===
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()

class_names = ["no_defected", "defected"]

print("Press Q in the preview window to exit.")
while True:
    frame = picam2.capture_array()

    # Convert RGBA to RGB if needed
    if frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Rotate by 135 degrees
    rotated_frame = rotate_image(frame, 135)

    # Preprocess for model
    input_img = cv2.resize(rotated_frame, (width, height))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = output[0][0]
    label = class_names[int(pred > 0.5)]
    conf = pred if pred > 0.5 else 1 - pred

    # Annotate and display
    color = (0, 255, 0) if label == "no_defected" else (0, 0, 255)
    text = f"{label} ({conf:.2f})"
    cv2.putText(rotated_frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    rotated_frame_bgr = cv2.cvtColor(rotated_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera", rotated_frame_bgr)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

picam2.stop()
cv2.destroyAllWindows()
