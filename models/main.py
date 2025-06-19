from picamera2 import Picamera2
import cv2
import time

# Initialize camera with optimized configuration
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "BGR888"},
    controls={"FrameRate": 30}
)
picam2.configure(config)
picam2.start()
time.sleep(1)  # Allow camera to initialize

try:
    while True:
        frame = picam2.capture_array()
        
        # Check if frame is valid
        if frame is not None and frame.size > 0:
            cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
