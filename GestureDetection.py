import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


def save_result(result, output_image, timestamp_ms):
    global value
    
    if not result.gestures:
        return
    
    if result.gestures[0][0].category_name != value:
        value = result.gestures[0][0].category_name
        print("Gesture:", value)
        

value = None  # track last gesture to avoid repeats
# Load model
BASE = python.BaseOptions("gesture_recognizer.task")
OPTIONS = vision.GestureRecognizerOptions(
    base_options=BASE,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=save_result
)
RECOGNIZER = vision.GestureRecognizer.create_from_options(OPTIONS)

# Set up camera
TITLE_CAPTURE = "Capture"
CAP = cv.VideoCapture(0)
CAP.set(cv.CAP_PROP_FRAME_WIDTH, 320)
CAP.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
#cv.namedWindow(TITLE_CAPTURE, cv.WINDOW_AUTOSIZE)

frame_counter = 0
process_every_n = CAP.get(cv.CAP_PROP_FPS) // 2
print(process_every_n)
# Frame loop
while True:
    ret, image = CAP.read()
    if not ret:
        break

    frame_counter += 1
    #cv.imshow(TITLE_CAPTURE, image)
    
    if frame_counter % process_every_n == 0:
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        timestamp = int(time.time() * 1000)
        RECOGNIZER.recognize_async(mp_image, timestamp)

    key = cv.waitKey(10)
    if key == ord('q'):
        break

RECOGNIZER.close()
CAP.release()
cv.destroyAllWindows()
