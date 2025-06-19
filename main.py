import cv2
import numpy as np
from picamera2 import Picamera2
import libcamera
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="model/model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # e.g. [1, 128, 128, 3]
height, width = input_shape[1], input_shape[2]

# === 2. Initialize camera with 180 degree flip ===
picam2 = Picamera2()
config = picam2.create_preview_configuration()
config["transform"] = libcamera.Transform(hflip=1, vflip=1)  # 180° flip
picam2.configure(config)
picam2.start()

class_names = ["no_defected", "defected"]

print("Press Q in the preview window to exit.")
while True:
    frame = picam2.capture_array()

    # Convert RGBA to RGB if needed
    if frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Preprocess for model
    input_img = cv2.resize(frame, (width, height))
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
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera", frame_bgr)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

picam2.stop()
cv2.destroyAllWindows()
