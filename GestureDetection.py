import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


def save_result(result, output_image, timestamp_ms):
    try:
        print("Recognition result:", result.gestures[0][0].category_name)
    except Exception as e:
        print("Callback error:", e)


BASE = python.BaseOptions("gesture_recognizer.task")
OPTIONS = vision.GestureRecognizerOptions(
    base_options=BASE,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=save_result
)
RECOGNIZER = vision.GestureRecognizer.create_from_options(OPTIONS)

TITLE_CAPTURE = "Capture"
CAP = cv.VideoCapture(0)
CAP.set(cv.CAP_PROP_FRAME_WIDTH, 256)
CAP.set(cv.CAP_PROP_FRAME_HEIGHT, 256)

while True:
    ret, image = CAP.read()
    if ret:
        cv.namedWindow(TITLE_CAPTURE, cv.WINDOW_AUTOSIZE)
        cv.imshow(TITLE_CAPTURE, image)

        timestamp = int(time.time() * 1000000) 
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        RECOGNIZER.recognize_async(mp_image, timestamp)

    key = cv.waitKey(10)
    if key == ord("q"):
        break

cv.destroyAllWindows()