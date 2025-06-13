import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

value = None  # track last gesture to avoid repeats

def save_result(result, output_image, timestamp_ms):
    global value
    try:
        if result.gestures and result.gestures[0][0].category_name != value:
            value = result.gestures[0][0].category_name
            print("Gesture:", value)
    except Exception as e:
        pass

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
cv.namedWindow(TITLE_CAPTURE, cv.WINDOW_AUTOSIZE)

# Frame loop
while True:
    ret, image = CAP.read()
    if not ret:
        break

    cv.imshow(TITLE_CAPTURE, image)

    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    timestamp = int(time.time() * 1000)
    RECOGNIZER.recognize_async(mp_image, timestamp)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

CAP.release()
cv.destroyAllWindows()
