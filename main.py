from picamera2 import Picamera2
import libcamera
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="../models/model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize camera with 180° flip
picam2 = Picamera2()
config = picam2.create_preview_configuration()
config["transform"] = libcamera.Transform(hflip=1, vflip=1)
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()
    # Resize and preprocess the frame for model input
    input_img = cv2.resize(frame, (224, 224))
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("Prediction:", output_data)

    # Show live preview
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
