import paho.mqtt.publish as publish
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math


def get_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def save_result(result, output_image, timestamp_ms):
    global HOST
    global value

    if not result.gestures:
        value["category"] = "none"
        value["landmark"] = (0, 0, 0)
        publish.single("gesture", value["category"], 2, hostname=HOST)
        return

    category = result.gestures[0][0].category_name
    if not category or category == "none":
        value["category"] = "none"
        value["landmark"] = (0, 0, 0)
        publish.single("gesture", value["category"], 2, hostname=HOST)
        return

    landmark = (
        result.hand_landmarks[0][0].x,
        result.hand_landmarks[0][0].y,
        result.hand_landmarks[0][0].z
    )
    if category != value["category"]:
        value["category"] = category
        value["landmark"] = landmark
        print("Gesture:", category)
        publish.single("gesture", value["category"], 2, hostname=HOST)
        return
        

HOST = "localhost"
BASE = python.BaseOptions("gesture_recognizer.task")
OPTIONS = vision.GestureRecognizerOptions(
    base_options=BASE,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=save_result
)
RECOGNIZER = vision.GestureRecognizer.create_from_options(OPTIONS)
value = {
    "category": "none",
    "landmark": (0, 0, 0)
}


def main():
    CAP = cv.VideoCapture(0)
    CAP.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    CAP.set(cv.CAP_PROP_FRAME_HEIGHT, 240)

    frame_counter = 0
    PROCESS_EVERY_N = CAP.get(cv.CAP_PROP_FPS) // 2

    while True:
        ret, image = CAP.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % PROCESS_EVERY_N == 0:
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            timestamp = int(time.time() * 1000)
            RECOGNIZER.recognize_async(mp_image, timestamp)
            frame_counter = 0

    RECOGNIZER.close()
    CAP.release()


if __name__ == "__main__":
    main()
